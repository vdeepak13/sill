
#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/operations.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/record_conversions.hpp>
#include <sill/learning/parameter_grid.hpp>
#include <sill/math/constants.hpp>
#include <sill/math/free_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Public methods: Constructors, getters, helpers
  // =========================================================================

  gaussian_crf_factor::gaussian_crf_factor()
    : base(), fixed_records_(false) { }

  gaussian_crf_factor::
  gaussian_crf_factor(const forward_range<vector_variable*>& Y_,
                      const forward_range<vector_variable*>& X_)
    : base(make_domain(Y_),
           copy_ptr<vector_domain>
           (new vector_domain(make_domain<vector_variable>(X_)))),
      ov(optimization_vector::size_type(vector_size(Y_),vector_size(X_)), 0),
      Y_(Y_.begin(), Y_.end()), X_(X_.begin(), X_.end()),
      fixed_records_(false), conditioned_f(Y_, 0.) {
    ov.zeros();
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const forward_range<vector_variable*>& Y_,
                      copy_ptr<vector_domain>& Xdomain_ptr_)
    : base(make_domain(Y_), Xdomain_ptr_),
      ov(optimization_vector::size_type(vector_size(Y_),vector_size(X_)), 0),
      Y_(Y_.begin(), Y_.end()), X_(X_.begin(), X_.end()),
      fixed_records_(false), conditioned_f(Y_, 0.) {
    ov.zeros();
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const optimization_vector& ov,
                      const vector_var_vector& Y_,
                      const vector_var_vector& X_)
    : base(make_domain(Y_),
           copy_ptr<vector_domain>(new vector_domain(make_domain(X_)))),
      ov(ov), Y_(Y_), X_(X_), fixed_records_(false), conditioned_f(Y_, 0.) {
    if (!ov.valid_size())
      throw std::invalid_argument
        ("gaussian_crf_factor constructor: ov dimensions do not match each other.");
    if ((ov.A.size1() != Y_.size()) || (ov.C.size2() != X_.size()))
      throw std::invalid_argument
        ("gaussian_crf_factor constructor: ov dimensions do not match Y,X.");
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const linear_regression& lr, const dataset& ds)
    : base(make_domain(Y_),
           copy_ptr<vector_domain>(new vector_domain(make_domain(X_)))),
      Y_(lr.Yvector()), X_(lr.Xvector()), fixed_records_(false),
      conditioned_f(Y_, 0.) {
    assert(Y_.size() > 0);
    mat ds_cov;
    if (vector_size(Y_) > 1) {
      ds.covariance(ds_cov, Y_);
      mat inv_cov;
      bool result = inv(ds_cov, inv_cov);
      if (!result) {
        throw std::runtime_error
          ("Matrix inverse failed in the construction of a gaussian_crf_factor from a linear_regression.");
      }
      result = chol(inv_cov, ov.A);
      if (!result) {
        throw std::runtime_error
          ("Cholesky decomposition failed in the construction of a gaussian_crf_factor from a linear_regression.");
      }
      ds_cov *= ov.A.transpose();
      ov.b = ds_cov * lr.weights().b;
      ov.C = ds_cov * lr.weights().A;
    } else {
      ov.A = identity(1);
      ov.b = lr.weights().b;
      ov.C = lr.weights().A;
    }
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const moment_gaussian& mg)
    : base(make_domain(mg.head()),
           copy_ptr<vector_domain>
           (new vector_domain(make_domain(mg.tail())))),
      Y_(mg.head()), X_(mg.tail()), fixed_records_(false),
      conditioned_f(Y_, 0.) {
    mat inv_cov;
    bool result = inv(mg.covariance(), inv_cov);
    if (!result) {
      throw inv_error("Matrix inverse failed in the construction of a gaussian_crf_factor from a moment_gaussian.");
    }
    result = chol(inv_cov, ov.A);
    if (!result) {
      throw chol_error("Cholesky decomposition failed in the construction of a gaussian_crf_factor from a moment_gaussian.");
    }
    ov.b = ov.A * mg.mean();
    if (mg.coefficients().size() > 0)
      ov.C = ov.A * mg.coefficients();
  }

  gaussian_crf_factor::gaussian_crf_factor(const canonical_gaussian& cg)
    : base(cg.arguments(), copy_ptr<vector_domain>(new vector_domain())),
      Y_(cg.argument_list()), X_(), fixed_records_(false), conditioned_f(cg) {
    bool result = chol(cg.inf_matrix(), ov.A);
    if (!result) {
      std::cerr << "Could not take Cholesky decomposition of lambda = \n"
                << cg.inf_matrix() << std::endl;
      throw chol_error("Cholesky decomposition failed in the construction of a gaussian_crf_factor from a canonical_gaussian.");
    }
    mat tmpmat;
    result = inv(ov.A * ov.A.transpose(), tmpmat);
    if (!result) {
      throw inv_error("Matrix inverse failed in the construction of a gaussian_crf_factor from a canonical_gaussian.");
    }
    ov.b = tmpmat * (ov.A * cg.inf_vector());
    // ov.C is empty
  }

  gaussian_crf_factor::gaussian_crf_factor(const constant_factor& cf)
    : base(), fixed_records_(false), conditioned_f(cf) { }

  const vector_var_vector& gaussian_crf_factor::output_arg_list() const {
    return Y_;
  }

  const vector_var_vector& gaussian_crf_factor::input_arg_list() const {
    return X_;
  }

  void gaussian_crf_factor::
  print(std::ostream& out, bool print_Y, bool print_X, bool print_vals) const {
    out << "F[";
    if (print_Y)
      out << Y_;
    else
      out << "*";
    out << ", ";
    if (print_X)
      out << X_;
    else
      out << "*";
    out << "]\n";
    if (print_vals)
      ov.print(out);
  }

  moment_gaussian gaussian_crf_factor::get_gaussian() const {
    mat sigma;
    bool result(ls_solve_chol(ov.A.transpose() * ov.A, identity(ov.A.size1()),
                              sigma));
    if (!result) {
      throw std::runtime_error
        ("Cholesky decomposition failed in gaussian_crf_factor::get_gaussian");
    }
    mat sigma_At(sigma * ov.A.transpose());
    mat mg_coeff;
    if (ov.C.size() > 0)
      mg_coeff = sigma_At * ov.C;
    return moment_gaussian(Y_, sigma_At * ov.b, sigma, X_, mg_coeff);
  }

  void
  gaussian_crf_factor::relabel_outputs_inputs(const output_domain_type& new_Y,
                                              const input_domain_type& new_X) {
    throw std::runtime_error
      ("gaussian_crf_factor::relabel_outputs_inputs NOT YET IMPLEMENTED!");
  }

  // Public methods: Probabilistic queries
  // =========================================================================

  double gaussian_crf_factor::v(const vector_assignment& a) const {
    return exp(logv(a));
  }

  double gaussian_crf_factor::v(const vector_record& r) const {
    return exp(logv(r));
  }

  double gaussian_crf_factor::logv(const assignment_type& a) const {
    vec y(ov.C.size1(), 0);
    vec x(ov.C.size2(), 0);
    vector_assignment2vector(a, Y_, y);
    vector_assignment2vector(a, X_, x);
    y = (ov.A * y) - ov.b - (ov.C * x);
    return (-.5) * inner_prod(y, y);
  }

  double gaussian_crf_factor::logv(const record_type& r) const {
    vec y(ov.C.size1(), 0);
    vec x(ov.C.size2(), 0);
    get_yx_values(r, y, x);
    y = (ov.A * y) - ov.b - (ov.C * x);
    return (-.5) * inner_prod(y, y);
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const vector_assignment& a) const {
    vec x(ov.C.size2(), 0);
    vector_assignment2vector(a, X_, x);
    return condition(x);
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const vector_record& r) const {
    vec x(ov.C.size2(), 0);
    get_x_values(r, x);
    return condition(x);
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const vec& x) const {
    if (x.size() != ov.C.size2()) {
      throw std::invalid_argument
        ("gaussian_crf_factor::condition(x) given x of size " +
         to_string(x.size()) + " but expected size " + to_string(ov.C.size2()));
    }
    if (Y_.size() == 0) // If this is a constant factor
      return conditioned_f;
    if (conditioned_f.argument_list() == Y_) { // avoid reallocation
      conditioned_f.inf_matrix() = ov.A.transpose() * ov.A;
      if (x.size() == 0) {
        conditioned_f.inf_vector() = ov.A.transpose() * ov.b;
      } else {
        conditioned_f.inf_vector() = ov.A.transpose() * (ov.b + ov.C * x);
      }
      conditioned_f.log_multiplier() = 0;
    } else {
      if (x.size() == 0) {
        conditioned_f.reset(Y_, ov.A.transpose() * ov.A,
                            ov.A.transpose() * ov.b);
      } else {
        conditioned_f.reset(Y_, ov.A.transpose() * ov.A,
                            ov.A.transpose() * (ov.b + ov.C * x));
      }
    }
    return conditioned_f;
  }

  gaussian_crf_factor&
  gaussian_crf_factor::
  partial_expectation_in_log_space(const output_domain_type& Y_part) {
    throw std::runtime_error("gaussian_crf_factor::partial_expectation_in_log_space NOT YET IMPLEMENTED!");
  }

  gaussian_crf_factor&
  gaussian_crf_factor::
  partial_expectation_in_log_space(const output_domain_type& Y_part,
                                   const dataset& ds) {
    throw std::runtime_error("gaussian_crf_factor::partial_expectation_in_log_space NOT YET IMPLEMENTED!");
  }

  gaussian_crf_factor&
  gaussian_crf_factor::marginalize_out(const output_domain_type& Y_other) {
    throw std::runtime_error
      ("gaussian_crf_factor::marginalize_out NOT YET IMPLEMENTED!");
  }

  gaussian_crf_factor&
  gaussian_crf_factor::partial_condition(const assignment_type& a,
                                         const output_domain_type& Y_part,
                                         const input_domain_type& X_part) {
    throw std::runtime_error
      ("gaussian_crf_factor::partial_condition NOT YET IMPLEMENTED!");
  }

  gaussian_crf_factor&
  gaussian_crf_factor::partial_condition(const record_type& r,
                                         const output_domain_type& Y_part,
                                         const input_domain_type& X_part) {
    throw std::runtime_error
      ("gaussian_crf_factor::partial_condition NOT YET IMPLEMENTED!");
  }

  double gaussian_crf_factor::log_expected_value(const dataset& ds) const {
    double val(0.);
    double total_ds_weight(0);
    size_t i(0);
    foreach(const vector_record& r, ds.records()) {
      vec y(ov.C.size1(), 0);
      vec x(ov.C.size2(), 0);
      get_yx_values(r, y, x);
      y = (ov.A * y) - ov.b - (ov.C * x);
      val += ds.weight(i) * (-.5) * inner_prod(y, y);
      total_ds_weight += ds.weight(i);
      ++i;
    }
    assert(total_ds_weight > 0);
    return (val / total_ds_weight);
  }

  gaussian_crf_factor&
  gaussian_crf_factor::combine_in(const gaussian_crf_factor& other, op_type op){
    if (arguments().size() > 0) {
      throw std::runtime_error
        ("gaussian_crf_factor::combine_in NOT YET FULLY IMPLEMENTED!");
    }
    switch (op) {
    case no_op:
      break;
    case sum_op:
    case minus_op:
    case product_op:
      throw std::runtime_error
        ("gaussian_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
      break;
    case divides_op:
      {
        double myval = this->v(vector_assignment());
        this->operator=(other);
        ov.reciprocal();
        ov *= myval;
      }
      break;
    case max_op:
    case min_op:
    case and_op:
    case or_op:
      throw std::runtime_error
        ("gaussian_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
      break;
    default:
      assert(false);
    }
    return *this;
  }

  gaussian_crf_factor&
  gaussian_crf_factor::combine_in(const constant_factor& other, op_type op) {
    throw std::runtime_error
      ("gaussian_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
  }

  /*
  gaussian_crf_factor&
  gaussian_crf_factor::combine_in_left(const constant_factor& cf, op_type op) {
    switch (op) {
    case no_op:
      break;
    case sum_op:
    case minus_op:
    case product_op:
      throw std::runtime_error
        ("gaussian_crf_factor::combine_in_left NOT FULLY IMPLEMENTED!");
      break;
    case divides_op:
      ov.reciprocal();  // Is this correct?
      ov *= cf;
      break;
    case max_op:
    case min_op:
    case and_op:
    case or_op:
      throw std::runtime_error
        ("gaussian_crf_factor::combine_in_left NOT FULLY IMPLEMENTED!");
      break;
    default:
      assert(false);
    }
    return *this;
  }
  */

  // Public: Learning methods from learnable_crf_factor interface
  // =========================================================================

  void gaussian_crf_factor::add_gradient
  (gaussian_crf_factor::optimization_vector& grad,
   const vector_record& r, double w) const {
    vec y(ov.A.size1(), 0);
    if (y.size() == 0)
      return;
    vec x(ov.C.size2(), 0);
    get_yx_values(r, y, x);
    vec tmpvec(ov.b);
    if (x.size() != 0)
      tmpvec += ov.C * x;
    tmpvec -= ov.A * y;
    tmpvec *= w;
    grad.A += outer_product(tmpvec, y);
    grad.b -= tmpvec;
    if (x.size() != 0)
      grad.C -= outer_product(tmpvec, x);
  }

  void gaussian_crf_factor::add_expected_gradient(optimization_vector& grad,
                                                  const vector_record& r,
                                                  const canonical_gaussian& fy,
                                                  double w) const {
    add_expected_gradient(grad, r, moment_gaussian(fy), w);
  }

  void
  gaussian_crf_factor::add_expected_gradient(optimization_vector& grad,
                                             const vector_record& r,
                                             const moment_gaussian& fy,
                                             double w) const {
    vec mu(fy.mean(Y_));
    if (mu.size() == 0)
      return;
    vec x(ov.C.size2(), 0);
    get_x_values(r, x);
    vec tmpvec(ov.b);
    if (x.size() != 0)
      tmpvec += ov.C * x;

    grad.A += outer_product(tmpvec, w * mu);
    grad.A -= w * ov.A * (fy.covariance(Y_) + outer_product(mu, mu));
    tmpvec -= ov.A * mu;
    tmpvec *= w;
    grad.b -= tmpvec;
    if (x.size() != 0)
      grad.C -= outer_product(tmpvec, x);
  }

  void
  gaussian_crf_factor::
  add_combined_gradient(optimization_vector& grad, const vector_record& r,
                        const canonical_gaussian& fy, double w) const {
    add_combined_gradient(grad, r, moment_gaussian(fy), w);
  }

  void
  gaussian_crf_factor::add_combined_gradient
  (optimization_vector& grad, const vector_record& r,
   const moment_gaussian& fy, double w) const {
    vec y(ov.A.size1(), 0);
    if (y.size() == 0)
      return;
    vec x(ov.C.size2(), 0);
    get_yx_values(r, y, x);
    vec mu(fy.mean(Y_));

    vec tmpvec(ov.b);
    if (x.size() != 0)
      tmpvec += ov.C * x;
    vec tmpvec2(tmpvec);
    tmpvec -= ov.A * y;
    tmpvec *= w;
    grad.A += outer_product(tmpvec, y);
    grad.b -= tmpvec;

    tmpvec2 -= ov.A * mu;
    tmpvec2 *= (-1. * w);
    grad.A += outer_product(tmpvec2, mu);
    grad.A += w * ov.A * fy.covariance(Y_);
    grad.b -= tmpvec2;
    if (x.size() != 0)
      grad.C -= outer_product(tmpvec + tmpvec2, x);
  }

  void gaussian_crf_factor::
  add_hessian_diag(optimization_vector& hessian, const vector_record& r,
                   double w) const {
    vec tmpvec(ov.A.size1());
    if (tmpvec.size() != 0) {
      get_y_values(r, tmpvec);
      elem_mult_inplace(tmpvec, tmpvec);
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.A.size2(); ++j)
        hessian.A.set_col(j, hessian.A.get_col(j) - tmpvec[j]);
    }
    hessian.b -= w;
    if (ov.C.size2() != 0) {
      tmpvec.resize(ov.C.size2());
      get_x_values(r, tmpvec);
      elem_mult_inplace(tmpvec, tmpvec);
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.C.size2(); ++j)
        hessian.C.set_col(j, hessian.C.get_col(j) - tmpvec[j]);
    }
  }

  void gaussian_crf_factor::
  add_expected_hessian_diag(optimization_vector& hessian,
                            const vector_record& r,
                            const canonical_gaussian& fy, double w) const {
    add_expected_hessian_diag(hessian, r, moment_gaussian(fy), w);
  }

  void gaussian_crf_factor::
  add_expected_hessian_diag(optimization_vector& hessian,
                            const vector_record& r,
                            const moment_gaussian& fy, double w) const {
    vec tmpvec(fy.mean(Y_));
    if (tmpvec.size() != 0) {
      elem_mult_inplace(tmpvec, tmpvec);
      tmpvec += fy.covariance_diag(Y_);
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.A.size2(); ++j)
        hessian.A.set_col(j, hessian.A.get_col(j) - tmpvec[j]);
    }
    hessian.b -= w;
    if (ov.C.size2() != 0) {
      tmpvec.resize(ov.C.size2());
      get_x_values(r, tmpvec);
      elem_mult_inplace(tmpvec, tmpvec);
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.C.size2(); ++j)
        hessian.C.set_col(j, hessian.C.get_col(j) - tmpvec[j]);
    }
  }

  void gaussian_crf_factor::
  add_expected_squared_gradient(optimization_vector& sqrgrad,
                                const vector_record& r,
                                const canonical_gaussian& fy, double w) const{
    add_expected_squared_gradient(sqrgrad, r, moment_gaussian(fy), w);
  }

  void gaussian_crf_factor::
  add_expected_squared_gradient(optimization_vector& sqrgrad,
                                const vector_record& r,
                                const moment_gaussian& fy, double w) const {
    vec mu(fy.mean(Y_));
    if (mu.size() == 0)
      return;
    mat cov(fy.covariance(Y_));
    vec x(ov.C.size2(), 0);
    get_x_values(r, x);
    mat tmpmat(outer_product(mu, mu));
    tmpmat += cov;
    vec Gdiag(diag(tmpmat));   // Gdiag(j) = G_{jj}
    tmpmat *= ov.A.transpose();  // (tmpmat = GA' now)
    vec b_Cx(ov.b);
    if (x.size() != 0)
      b_Cx += ov.C * x;        // b_Cx(i) = b_i + C_{i.} \cdot x
    vec A_mu(ov.A * mu);       // A_mu(i) = A_{i.} \cdot \mu
    vec tmpvec(b_Cx);
    tmpvec *= -2.;
    tmpvec += A_mu;
    mat tmpmat2(outer_product(tmpvec, mu));
    tmpmat2 += tmpmat.transpose();
    tmpmat2 *= (2. * w);
    mat A_sigma(ov.A);
    A_sigma *= cov;            // A_sigma(i,j) = A_{i.} \cdot \sigma_{j.}
    elem_mult_inplace(A_sigma, tmpmat2);
    sqrgrad.A += tmpmat2;
    tmpvec = A_mu;
    tmpvec *= -2.;
    tmpvec += b_Cx;
    elem_mult_inplace(b_Cx, tmpvec);
    elem_mult_inplace(ov.A.transpose(), tmpmat);
    tmpvec += sum(tmpmat, 2);
    if (w != 1)
      tmpvec *= w;
    sqrgrad.A += outer_product(tmpvec, Gdiag);
    sqrgrad.b += tmpvec;
    if (x.size() != 0) {
      elem_mult_inplace(x, x);
      sqrgrad.C += outer_product(tmpvec, x);
    }
  }

  double gaussian_crf_factor::regularization_penalty
  (const regularization_type& reg) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0: // none
      return 0.;
    case 2: // L2 on b,C,A
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -=
            reg.lambdas[0] * (ov.C.inner_prod(ov.C) + ov.b.inner_prod(ov.b));
        if (reg.lambdas[1] != 0)
          val -= reg.lambdas[1] * ov.A.inner_prod(ov.A);
        return (.5 * val);
      }
    case 3: // L2 on b,C and logdet((A'A)^-1)
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -= .5 * reg.lambdas[0]
            * (ov.C.inner_prod(ov.C) + ov.b.inner_prod(ov.b));
//        val += (reg.lambdas[1] + 2 * ov.A.size1()) * logdet(ov.A);
        val += (.5 * reg.lambdas[1] + ov.A.size1()) * logdet(ov.A.transpose() * ov.A);
        return val;
      }
    case 4: // L2 on b,C and [ logdet((A'A)^-1) + tr(A'A) ]
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -= .5 * reg.lambdas[0]
            * (ov.C.inner_prod(ov.C) + ov.b.inner_prod(ov.b));
//        val += (reg.lambdas[1] + 2 * ov.A.size1()) * logdet(ov.A);
        val += (.5 * reg.lambdas[1] + ov.A.size1()) * logdet(ov.A.transpose() * ov.A);
        if (reg.lambdas[1] != 0)
          val -= .5 * reg.lambdas[1] * ov.A.inner_prod(ov.A);
        return val;
      }
    case 5: // L2 on b,C and tr((A'A)^-1)
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -=
            reg.lambdas[0] * (ov.C.inner_prod(ov.C) + ov.b.inner_prod(ov.b));
        if (reg.lambdas[1] != 0) {
          vec eig_AtA;
          bool result = eig_sym(ov.A.transpose() * ov.A, eig_AtA);
          if (!result) {
            throw std::runtime_error("eig_sym failed in gaussian_crf_factor::regularization_penalty().");
          }
          val -= sum(elem_div(reg.lambdas[1], eig_AtA));
        }
        return (.5 * val);
      }
    case 6: // L2 on b,C and [ -logdet((A'A)^-1) + tr((A'A)^-1) ]
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -= .5 * reg.lambdas[0]
            * (ov.C.inner_prod(ov.C) + ov.b.inner_prod(ov.b));
        if (reg.lambdas[1] != 0) {
//          val -= reg.lambdas[1] * logdet(ov.A);
          val -= (.5 * reg.lambdas[1]) * logdet(ov.A.transpose() * ov.A);
          vec eig_AtA;
          bool result = eig_sym(ov.A.transpose() * ov.A, eig_AtA);
          if (!result) {
            throw std::runtime_error("eig_sym failed in gaussian_crf_factor::regularization_penalty().");
          }
          val -= .5 * sum(elem_div(reg.lambdas[1], eig_AtA));
        }
        return val;
      }
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  void gaussian_crf_factor::add_regularization_gradient
  (optimization_vector& grad, const regularization_type& reg, double w) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0: // none
      return;
    case 2: // L2 on b,C,A
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      if (reg.lambdas[1] != 0)
        grad.A -= w * reg.lambdas[1] * ov.A;
      return;
    case 3: // L2 on b,C and logdet((A'A)^-1)
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      {
        mat AtA_inv_At;
        bool result = ls_solve_chol(ov.A.transpose() * ov.A, ov.A.transpose(),
                                    AtA_inv_At);
        if (!result) {
          throw ls_solve_chol_error("ls_solve_chol failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A -=
          w * (reg.lambdas[1] + 2 * ov.A.size1()) * AtA_inv_At.transpose();
      }
      return;
    case 4: // L2 on b,C and [ logdet((A'A)^-1) + tr(A'A) ]
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      {
        mat AtA_inv;
        bool result = inv(ov.A.transpose() * ov.A, AtA_inv);
        if (!result) {
          throw inv_error("Matrix inverse failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A -= ov.A * w * ((reg.lambdas[1] + 2 * ov.A.size1()) * AtA_inv
                              + reg.lambdas[1] * identity(AtA_inv.size1()));
      }
      return;
    case 5: // L2 on b,C and tr((A'A)^-1)
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      if (reg.lambdas[1] != 0) {
        mat tmpmat(ov.A.transpose());
        tmpmat *= ov.A;
        bool result = ls_solve_chol(tmpmat * tmpmat, ov.A.transpose(), tmpmat);
        if (!result) {
          throw ls_solve_chol_error("ls_solve_chol failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A += w * reg.lambdas[1] * tmpmat.transpose();
      }
      return;
    case 6: // L2 on b,C and [ -logdet((A'A)^-1) + tr((A'A)^-1) ]
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      if (reg.lambdas[1] != 0) {
        mat AtA_inv;
        bool result = inv(ov.A.transpose() * ov.A, AtA_inv);
        if (!result) {
          throw inv_error("Matrix inverse failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A += w * reg.lambdas[1]
          * ov.A * AtA_inv * (AtA_inv - identity(ov.A.size1()));
      }
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  void gaussian_crf_factor::
  add_regularization_hessian_diag(optimization_vector& hd,
                                  const regularization_type& reg,
                                  double w) const {
    // TO DO: ADD SUPPORT FOR THE OTHER TYPES OF REGULARIZATION
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0: // none
      return;
    case 2: // L2 on b,C,A
      if (reg.lambdas[0] != 0) {
        hd.C -= w * reg.lambdas[0];
        hd.b -= w * reg.lambdas[0];
      }
      if (reg.lambdas[1] != 0)
        hd.A -= w * reg.lambdas[1];
      return;
    case 3: // L2 on b,C and logdet((A'A)^-1)
    case 4: // L2 on b,C and [ logdet((A'A)^-1) + tr(A'A) ]
    case 5: // L2 on b,C and tr((A'A)^-1)
    case 6: // L2 on b,C and [ -logdet((A'A)^-1) + tr((A'A)^-1) ]
      std::cerr << "GAUSSIAN_CRF_FACTOR::ADD_REGULARIZATION_HESSIAN_DIAG() NOT YET FULLY IMPLEMENTED!!" << std::endl;
      assert(false);
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

}  // namespace sill

#include <sill/macros_undef.hpp>
