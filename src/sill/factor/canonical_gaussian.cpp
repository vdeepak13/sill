#include <algorithm>

#include <sill/base/universe.hpp>
#include <sill/math/constants.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/operations.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Constructors, conversion, and initialization
  //============================================================================
  void canonical_gaussian::
  initialize(const forward_range<vector_variable*>& args, bool use_default) {
    // Check the matrix sizes
    arg_list.clear();
    arg_list.insert(arg_list.end(), args.begin(), args.end());
    size_t n = vector_size(arg_list);
    // Initialize the variable indices
    compute_indices(arg_list);

    if (use_default) {
      // Initialize the matrices
      lambda = zeros(n, n);
      eta = zeros(n);
      log_mult = 0;
    } else {
      assert(lambda.n_rows == n && lambda.n_cols == n);
      assert(eta.n_elem == n);
    }
  }

  canonical_gaussian::
  canonical_gaussian(const vector_domain& args)
    : gaussian_factor(args), log_mult(0) {
    initialize(make_vector(args), true);
  }

  canonical_gaussian::
  canonical_gaussian(const vector_domain& args,
                     double value)
    : gaussian_factor(args), log_mult(std::log(value)) {
    initialize(make_vector(args), true);
  }

  canonical_gaussian::
  canonical_gaussian(const vector_var_vector& args)
    : gaussian_factor(args), log_mult(0) {
    initialize(args, true);
  }

  canonical_gaussian::
  canonical_gaussian(const vector_var_vector& args, double value)
    : gaussian_factor(args), log_mult(std::log(value)) {
    initialize(args, true);
  }

  canonical_gaussian::
  canonical_gaussian(const forward_range<vector_variable*>& args, double value)
    : gaussian_factor(args), log_mult(std::log(value)) {
    initialize(args, true);
  }

  canonical_gaussian::
  canonical_gaussian(const vector_var_vector& args,
                     const mat& lambda,
                     const vec& eta,
                     double log_mult)
    : gaussian_factor(make_domain(args)), lambda(lambda), eta(eta),
      log_mult(log_mult) {
    // assert(symmetric(lambda));
    initialize(args, false);
  }


  canonical_gaussian::
  canonical_gaussian(const constant_factor& factor)
    : log_mult(std::log(factor.value)) { }

  canonical_gaussian::canonical_gaussian(const moment_gaussian& mg) {
    size_t nhead = mg.size_head();
    size_t ntail = mg.size_tail();
    size_t n = nhead + ntail;
    lambda.set_size(n, n);
    eta.set_size(n);

    // Initialize the argument list
    arg_list = concat(mg.head_list, mg.tail_list);
    args = vector_domain(arg_list.begin(), arg_list.end());

    // Initialize the argument map
    compute_indices(arg_list);

    // Initialize the matrices
    // TODO: deal with empty head or tail
    span ih(0, nhead-1);
    span it(nhead, n-1);
    mat invcov = inv(mg.cov);

    if (nhead > 0) {
      lambda(ih, ih) = invcov;
      eta(ih) = invcov * mg.cmean;
    }

    if (ntail > 0) {
      mat Atinvcov = mg.coeff.transpose()*invcov;
      lambda(it, it) = Atinvcov*mg.coeff;
      lambda(ih, it) = -trans(Atinvcov);
      lambda(it, ih) = -Atinvcov;
      eta(it) = -Atinvcov * mg.cmean;
    }

    log_mult = mg.likelihood.log_value()
      - .5 * (mg.cmean.size() * std::log(2*pi()) + log_det(mg.cov)
              + as_scalar(trans(mg.cmean) * invcov * mg.cmean);
  }

  /*
  canonical_gaussian&
  canonical_gaussian::operator=(const canonical_gaussian& other) {
    var_range = other.var_range;
    args = other.args;
    arg_list = other.arg_list;
    lambda = other.lambda;
    eta = other.eta;
    log_mult = other.log_mult;
    return *this;
  }
  */

  canonical_gaussian::operator constant_factor() const {
    assert(this->arguments().empty());
    return constant_factor(exp(log_mult));
  }

  canonical_gaussian::operator std::string() const {
    std::ostringstream out; out << *this; return out.str();
  }

  void canonical_gaussian::reset(const vector_var_vector& args,
                                 const mat& lambda, const vec& eta,
                                 double log_mult) {
    this->args.clear();
    this->args.insert(args.begin(), args.end());
    this->lambda = lambda;
    this->eta = eta;
    this->log_mult = log_mult;
    // assert(symmetric(lambda));
    var_range.clear();
    initialize(args, false);
  }

  // Serialization
  //============================================================================

  void canonical_gaussian::save(oarchive& ar) const {
    ar << arg_list << lambda << eta << log_mult;
  }

  void canonical_gaussian::load(iarchive& ar) {
    ar >> arg_list >> lambda >> eta >> log_mult;
    args = vector_domain(arg_list.begin(), arg_list.end());
    var_range.clear();
    compute_indices(arg_list);
  }

  // Accessors
  //==========================================================================

  const vector_var_vector& canonical_gaussian::argument_list() const {
    return arg_list;
  }

  size_t canonical_gaussian::size() const {
    return eta.n_elem;
  }

  const mat& canonical_gaussian::inf_matrix() const {
    return lambda;
  }
    
  mat& canonical_gaussian::inf_matrix() {
    return lambda;
  }

  const vec& canonical_gaussian::inf_vector() const { 
    return eta;
  }

  vec& canonical_gaussian::inf_vector() {
    return eta;
  }

  double canonical_gaussian::log_multiplier() const {
    return log_mult;
  }

  double& canonical_gaussian::log_multiplier() {
    return log_mult;
  }

  mat canonical_gaussian::inf_matrix(const vector_var_vector& args) const {
    uvec ind(indices(args));
    return lambda(ind, ind);
  }

  vec canonical_gaussian::inf_vector(const vector_var_vector& args) const {
    uvec ind(this->indices(args));
    return eta(ind);
  }

  // Comparison operators
  //============================================================================
  bool canonical_gaussian::operator==(const canonical_gaussian& other) const {
    return arguments() == other.arguments() &&
      inf_vector() == other.inf_vector(arg_list) &&
      inf_matrix() == other.inf_matrix(arg_list);
  }

  bool canonical_gaussian::operator!=(const canonical_gaussian& other) const {
    return !operator==(other);
  }

  bool canonical_gaussian::operator<(const canonical_gaussian& other) const {
    if (this->arguments() < other.arguments()) return true;

    if (this->arguments() == other.arguments()) {
      vec other_vector = other.inf_vector(arg_list);
      if (sill::lexicographical_compare(inf_vector(), other_vector))
        return true; // inf_vector() < other_vector

      if (inf_vector() == other_vector) {
        mat other_matrix = other.inf_matrix(arg_list);
        for(size_t i = 0; i<lambda.n_rows; i++) {
          for(size_t j = 0; j<lambda.n_cols; j++) {
            if (lambda(i,j) != other_matrix(i,j))
              return lambda(i,j) < other_matrix(i,j);
          }
        }
        return false; // (*this) == other
      }
    }
    return false;
  }

  // Factor operations
  //==========================================================================
  logarithmic<double>
  canonical_gaussian::operator()(const vector_assignment& a) const {
    return logarithmic<double>(logv(a), log_tag());
  }

  logarithmic<double>
  canonical_gaussian::operator()(const record_type& r) const {
    return logarithmic<double>(logv(r), log_tag());
  }

  double canonical_gaussian::logv(const vector_assignment& a) const {
    if (eta.is_empty())
      return 0; // shouldn't this be log_mult?
    vec v = sill::concat(values(a, arg_list));
    // will assertion if a does not cover the arguments of this
    return - 0.5*dot(v, lambda*v) + dot(v, eta) + log_mult;
  }

  double canonical_gaussian::logv(const record_type& r) const {
    if (eta.is_empty())
      return 0; // shouldn't this be log_mult?
    vec v = zeros(eta.n_elem);
    r.vector_values(v, arg_list);
    // will assertion if a does not cover the arguments of this
    return - 0.5*as_scalar(trans(v) * lambda * v) + dot(v, eta) + log_mult;
  }

  canonical_gaussian
  canonical_gaussian::collapse(op_type op, const vector_domain& retain) const {
    canonical_gaussian cg;
    collapse(op, retain, cg);
    return cg;
  }

  void canonical_gaussian::collapse(op_type op,
                                    const vector_domain& retain,
                                    canonical_gaussian& cg) const {
    collapse_(op, retain, true, cg);
  } // collapse(op, retain, cg)

  void
  canonical_gaussian::collapse_unnormalized(op_type op,
                                            const vector_domain& retain,
                                            canonical_gaussian& cg) const {
    collapse_(op, retain, false, cg);
  } // collapse_unnormalized

  canonical_gaussian
  canonical_gaussian::restrict(const vector_assignment& a) const {
    // Determine the retained (x) and the restricted variables (y)
    vector_var_vector x, y;
    foreach(vector_variable* v, arg_list) {
      if (a.count(v) == 0)
        x.push_back(v);
      else
        y.push_back(v);
    }
    // If the arguments of x are disjoint from the bound variables,
    // we can simply return a copy of the factor
    if (y.empty()) return *this;

    uvec ix = indices(x);
    uvec iy = indices(y);
    vec vy = sill::concat(values(a, y));
    assert(vy.n_elem == iy.n_elem);

    if (x.empty()) {
      return canonical_gaussian(log_mult + dot(eta(iy), vy)
                                - 0.5*as_scalar(trans(vy) * lambda(iy,iy) * vy));
    }
    else
      return canonical_gaussian
        (x,
         lambda(ix, ix),
         eta(ix) - lambda(ix, iy)*vy,
         log_mult + dot(eta(iy), vy)
         - 0.5 * as_scalar(trans(vy) * lambda(iy,iy) * vy));
  } // restrict(a)

  void canonical_gaussian::
  restrict(const record_type& r, const vector_domain& r_vars,
           canonical_gaussian& f) const {
    this->restrict(r, r_vars, false, f);
  }

  void canonical_gaussian::
  restrict(const record_type& r, const vector_domain& r_vars,
           bool strict, canonical_gaussian& f) const {
    // Determine the retained (x) and the restricted variables (y)
    vector_var_vector x, y;
    foreach(vector_variable* v, arg_list) {
      if (r_vars.count(v) == 0) {
        x.push_back(v);
      } else {
        if (!r.has_variable(v)) {
          if (strict) {
            throw std::invalid_argument
              (std::string("canonical_gaussian::restrict(r,r_vars,strict,f)") +
               " was given strict=true, but intersect(f.arguments(), r_vars)" +
               " contained a variable which did not appear in keys(r).");
          }
          x.push_back(v);
        } else {
          y.push_back(v);
        }
      }
    }

    // If the arguments of x are disjoint from the bound variables,
    // we can simply return a copy of the factor
    if (y.empty()) {
      f = *this;
      return;
    }

    uvec ix = indices(x);
    uvec iy = indices(y);
    vec vy;
    r.vector_values(vy, y);
    assert(vy.n_elem == iy.n_elem);

    if (x.empty()) {
      f = canonical_gaussian(log_mult + dot(eta(iy), vy)
                             - 0.5*as_scalar(trans(vy) * lambda(iy,iy) * vy));
    } else {
      f = canonical_gaussian(x,
                             lambda(ix, ix),
                             eta(ix) - lambda(ix, iy)*vy,
                             log_mult + dot(eta(iy), vy)
                             - 0.5*as_scalar(vy * lambda(iy,iy) * vy));
    }
  } // restrict(f, r, r_vars, strict)

  canonical_gaussian&
  canonical_gaussian::subst_args(const vector_var_map& map) {
    gaussian_factor::subst_args(map);

    foreach(vector_variable* &a, arg_list) {
      if (map.count(a)) {
        a = safe_get(map, a);
      }
    }
    return *this;
  }

  canonical_gaussian
  canonical_gaussian::marginal(const vector_domain& retain) const {
    return collapse(sum_op, retain);
  }

  void
  canonical_gaussian::
  marginal(canonical_gaussian& cg, const vector_domain& retain) const {
    collapse(sum_op, retain, cg);
  }

  canonical_gaussian
  canonical_gaussian::conditional(const vector_domain& B) const {
    assert(includes(arguments(), B));
    canonical_gaussian PB(marginal(B));
    return (*this) / PB;
  }

  bool canonical_gaussian::is_normalizable() const {
    if (lambda.size() == 0)
      return true;
    mat lambda_inv;
    bool result = inv(lambda, lambda_inv);
    return result;
  }

  double canonical_gaussian::norm_constant() const {
    return std::exp(log_norm_constant());
  }

  double canonical_gaussian::log_norm_constant() const {
    mat lambda_inv;
    if (lambda.is_empty())
      return 0;
    try {
      lambda_inv = inv(lambda);
    } catch(std::runtime_error& e) {
      if (lambda.n_elem <= 16)
        std::cerr << "Lambda:\n" << lambda << std::endl;
      throw invalid_operation("Inversion of lambda matrix failed in canonical_gaussian::log_norm_constant.");
    }
    return (-0.5 * (eta.n_elem * std::log(2*pi())
                    - log_det(lambda)
                    + as_scalar(trans(eta) * trans(lambda_inv) * eta)));
    // why trans(lambda_inv) ?
    /*
    return (-.5 * (eta.n_elem * std::log(2*pi())
                   - logdet(lambda)
                   + dot(eta, lambda * eta)));
    */
  }

  canonical_gaussian& canonical_gaussian::normalize() {
    log_mult = log_norm_constant();
    return *this;
  }

  canonical_gaussian
  canonical_gaussian::maximum(const vector_domain& retain) const {
    vector_assignment a;
    vector_assignment means(arg_max()); // This could be more efficient.
    foreach(vector_variable* v, arg_list)
      if (retain.count(v) == 0)
        a[v] = means[v];
    return restrict(a);
  }

  vector_assignment canonical_gaussian::arg_max() const {
    mat cov;
    if (lambda.size() == 0)
      return vector_assignment();
    bool result = inv(lambda, cov);
    if (!result) {
      std::cerr << "In canonical_gaussian::arg_max, could not invert lambda =\n"
                << lambda << std::endl;
      throw invalid_operation("The canonical_gaussian does not represent a valid marginal distribution.");
    }
    vec mu(cov * eta);
    return make_assignment(arg_list, mu);
  }

  double canonical_gaussian::entropy(double base) const {
    size_t N = eta.n_elem;
    return (N + ((N*std::log(2.0 * pi()) - logdet(lambda)) / std::log(base)))/2.0;
  }

  double canonical_gaussian::entropy() const {
    return entropy(e());
  }

  double canonical_gaussian::relative_entropy(const canonical_gaussian& q) const
  {
    assert(false); // not implemented yet
    return 0;
  }

  double canonical_gaussian::mutual_information(const vector_domain& d1,
                                                const vector_domain& d2) const {
    // TO DO: Make this more efficient.
    if (!set_disjoint(d1, d2))
      throw std::runtime_error
        ("canonical_gaussian::mutual_information() called for non-disjoint sets of variables.");
    if ((!includes(args, d1)) || (!includes(args, d2)))
      throw std::runtime_error
        ("canonical_gaussian::mutual_information() called with variables not in the factor arguments.");
    // I(d1; d2) = H(d1) + H(d2) - H(d1,d2)
    canonical_gaussian cg1(marginal(d1));
    canonical_gaussian cg2(marginal(d2));
    return cg1.entropy() + cg2.entropy() - entropy();
  }

  // Factor operations: combining factors
  //==========================================================================

  canonical_gaussian&
  canonical_gaussian::combine_in(const canonical_gaussian& x, op_type op) {
    check_supported(op, combine_ops);
    double sign = (op == product_op) ? 1 : -1;
    if (!includes(arguments(), x.arguments())) {
//       size_t n(eta.n_elem);
//       for (vector_domain::const_iterator it(x.args.begin());
//            it != x.args.end(); ++it) {
//         vector_variable* v = *it;
//         if (args.insert(v).second) { // then variable was not yet in args
//           var_range[v] = irange(n, n + v->size());
//           n += v->size();
//           arg_list.push_back(v);
//         }
//       }
//       lambda.resize(n, n, true);
//       eta.resize(n, true);
      *this = combine(*this, x, op);
      return *this;
    }
    uvec indx;
    indices(x.arg_list, indx);
    if (sign == 1) {
      lambda(indx, indx) += x.lambda;
      eta(indx) += x.eta;
      log_mult += x.log_mult;
    } else {
      lambda(indx, indx) -= x.lambda;
      eta(indx) -= x.eta;
      log_mult -= x.log_mult;
    }
    return *this;
  }

  // Other operations
  //==========================================================================
  bool canonical_gaussian::enforce_psd(const vec& mean) {
    assert(mean.size() == size());
    vec d;
    mat v;
    bool result = eig_sym(lambda, d, v);
    if (!result) {
      using namespace std;
      cerr << lambda << endl;
      assert(result);
    }
    double mine = min(d);
    if (mine < 0) {
//       std::cerr << "Information matrix not PSD (min eigenvalue " << mine
//                 << "); adjusting." << std::endl;
      for(size_t i = 0; i < d.size(); i++) 
        d(i) = std::max(0.0, d(i));
      mat new_lambda = v * diagmat(d) * trans(v);
      eta += (new_lambda - lambda) * mean;
      lambda = new_lambda;
      return false;
    } else {
      return true;
    }
  }

  void canonical_gaussian::collapse_(op_type op,
                                     const vector_domain& retain,
                                     bool renormalize,
                                     canonical_gaussian& cg) const {
    check_supported(op, collapse_ops);

    if (retain.empty()) {
      cg.var_range.clear();
      cg.args.clear();
      cg.arg_list.clear();
      cg.lambda.reset();
      cg.eta.reset();
      if (renormalize)
        cg.log_mult = log_mult - log_norm_constant();
      return;
    }
    vector_var_vector x, y; // retained, marginalized out
    foreach(vector_variable* v, arg_list) {
      if (retain.count(v) == 0)
        y.push_back(v);
      else
        x.push_back(v);
    }
    if (y.empty()) {
      if (cg.arg_list == x) {
        cg.lambda = lambda;
        cg.eta = eta;
        cg.log_mult = log_mult;
      } else {
        cg = *this;
      }
    } else {
      uvec ix = indices(x);
      uvec iy = indices(y);
      mat invyy_lamyx;
      //bool info = ls_solve_chol(lambda(iy,iy), lambda(iy,ix), invyy_lamyx);
      // Armadillo does not yet suppport solve using CHolesky decomposition
      // try solving via LU factorization
      // Note: ls_solve_chol failed on some symmetric matrices with
      //       positive determinants, so it's possible the IT++
      //       implementation is buggy.
      //       (ls_solve has worked in these cases so far.)
      bool info = solve(invyy_lamyx, lambda(iy,iy), lambda(iy,ix));
      // 
      if (!info) {
        if (iy.n_elem * ix.n_elem < 16 &&
            iy.n_elem * iy.n_elem < 16)
          std::cerr << "Lambda(iy,iy):\n" << lambda(iy,iy) << "\n"
                    << "Lambda(iy,ix):\n" << lambda(iy,ix) << std::endl;
        throw invalid_operation
          (std::string("canonical_gaussian::collapse:") +
           " LU factorization failed.");
      }
      double old_log_norm_constant = (renormalize ? log_norm_constant() : 0);
      if (cg.arg_list == x) {
        cg.lambda = lambda(ix,ix) - lambda(ix,iy) * invyy_lamyx;
        cg.eta = eta(ix) - invyy_lamyx.transpose() * eta(iy);
      } else {
        cg.reset(x,
                 lambda(ix,ix) - lambda(ix,iy) * invyy_lamyx,
                 eta(ix) - invyy_lamyx.transpose() * eta(iy));
      }
      if (renormalize) {
        cg.log_mult = log_mult - old_log_norm_constant + cg.log_norm_constant();
      }
    }
  } // collapse_

  // Free functions
  //==========================================================================
  canonical_gaussian combine(const canonical_gaussian& x,
                             const canonical_gaussian& y,
                             op_type op) {
    factor::check_supported(op, canonical_gaussian::combine_ops);
    double sign = (op == product_op) ? 1 : -1;

    vector_domain args = set_union(x.arguments(), y.arguments());
    canonical_gaussian result(args, 0);
    uvec indx = result.indices(x.arg_list);
    uvec indy = result.indices(y.arg_list);

    result.lambda(indx, indx)  = x.lambda;
    result.lambda(indy, indy) += sign*y.lambda;
    result.eta(indx)  = x.eta;
    result.eta(indy) += sign*y.eta;
    result.log_mult = x.log_mult + sign*y.log_mult;

    return result;
  }

  double norm_inf(const canonical_gaussian& x, const canonical_gaussian& y) {
    assert(x.arguments() == y.arguments());
    double vec_norm =
      max(x.inf_vector() - y.inf_vector(x.argument_list()));
    double mat_norm =
      mat(x.inf_matrix() - y.inf_matrix(x.argument_list())).max();
    return std::max(vec_norm, mat_norm);
  }

  canonical_gaussian pow(const canonical_gaussian& cg, double a) {
    return canonical_gaussian(cg.argument_list(),
                              cg.inf_matrix() * a,
                              cg.inf_vector() * a,
                              cg.log_multiplier() * a);
  }

  vector_assignment arg_max(const canonical_gaussian& cg) {
    return cg.arg_max();
  }

  canonical_gaussian weighted_update(const canonical_gaussian& f1,
                                     const canonical_gaussian& f2,
                                     double a) {
    if (a == 1)
      return f2;
    else if (a == 0)
      return f1;
    else
      return pow(f1, 1-a) * pow(f2, a);
  }

  std::ostream& operator<<(std::ostream& out, const canonical_gaussian& cg) {
    out << "#F(CG|" << cg.argument_list()
        << "|" << cg.inf_matrix()
        << "|" << cg.inf_vector()
        << "|" << cg.log_multiplier() << ")";
    return out;
  }

} // namespace sill

