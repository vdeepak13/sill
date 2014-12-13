
#include <sill/factor/crf/gaussian_crf_factor.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/learning/dataset_old/dataset_view.hpp>
#include <sill/learning/dataset_old/record_conversions.hpp>
#include <sill/learning/validation/parameter_grid.hpp>
#include <sill/math/constants.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Public methods: Constructors, getters, helpers
  // =========================================================================

  gaussian_crf_factor::gaussian_crf_factor()
    : base(), fixed_records_(false), relabeled(false) { }

  gaussian_crf_factor::
  gaussian_crf_factor(const forward_range<vector_variable*>& Y_,
                      const forward_range<vector_variable*>& X_)
    : base(make_domain(Y_),
           copy_ptr<vector_domain>
           (new vector_domain(make_domain<vector_variable>(X_)))),
      head_(Y_.begin(), Y_.end()), tail_(X_.begin(), X_.end()),
      ov(optimization_vector::size_type(vector_size(head_),vector_size(tail_)),
         0),
      fixed_records_(false),
      conditioned_f(vector_var_vector(Y_.begin(), Y_.end())),
      relabeled(false) {
    ov.zeros();
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const forward_range<vector_variable*>& Y_,
                      copy_ptr<vector_domain>& Xdomain_ptr_)
    : base(make_domain(Y_), Xdomain_ptr_),
      head_(Y_.begin(), Y_.end()),
      tail_(Xdomain_ptr_->begin(), Xdomain_ptr_->end()),
      ov(optimization_vector::size_type(vector_size(head_),vector_size(tail_)),
         0),
      fixed_records_(false),
      conditioned_f(vector_var_vector(Y_.begin(), Y_.end())),
      relabeled(false) {
    ov.zeros();
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const optimization_vector& ov,
                      const vector_var_vector& Y_,
                      const vector_var_vector& X_)
    : base(make_domain(Y_),
           copy_ptr<vector_domain>(new vector_domain(make_domain(X_)))),
      head_(Y_), tail_(X_), ov(ov), fixed_records_(false),
      conditioned_f(Y_), relabeled(false) {
    if (!ov.valid_size())
      throw std::invalid_argument
        (std::string("gaussian_crf_factor constructor:") +
         " ov dimensions do not match each other.");
    if (ov.A.n_rows != vector_size(Y_) || ov.C.n_cols != vector_size(X_))
      throw std::invalid_argument
        ("gaussian_crf_factor constructor: ov dimensions do not match Y,X.");
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const linear_regression& lr, const dataset<dense_linear_algebra<> >& ds)
    : base(make_domain(lr.Yvector()),
           copy_ptr<vector_domain>
           (new vector_domain(make_domain(lr.Xvector())))),
      head_(lr.Yvector()), tail_(lr.Xvector()),
      fixed_records_(false), conditioned_f(head_), relabeled(false) {
    assert(head_.size() > 0);
    mat ds_cov;
    if (vector_size(head_) > 1) {
      ds.covariance(ds_cov, head_);
      mat inv_cov;
      bool result = inv(inv_cov, ds_cov);
      if (!result) {
        throw std::runtime_error
          ("Matrix inverse failed in the construction of a gaussian_crf_factor from a linear_regression.");
      }
      result = chol(ov.A, inv_cov);
      if (!result) {
        throw std::runtime_error
          ("Cholesky decomposition failed in the construction of a gaussian_crf_factor from a linear_regression.");
      }
      ds_cov *= trans(ov.A);
      ov.b = ds_cov * lr.weights().b;
      ov.C = ds_cov * lr.weights().A;
    } else {
      ov.A = eye(1,1);
      ov.b = lr.weights().b;
      ov.C = lr.weights().A;
    }
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const moment_gaussian& mg)
    : base(make_domain(mg.head()),
           copy_ptr<vector_domain>
           (new vector_domain(make_domain(mg.tail())))),
      head_(mg.head()), tail_(mg.tail()), fixed_records_(false),
      conditioned_f(head_), relabeled(false) {
    mat inv_cov;
    bool result = inv(inv_cov, mg.covariance());
    if (!result) {
      throw inv_error("Matrix inverse failed in the construction of a gaussian_crf_factor from a moment_gaussian.");
    }
    result = chol(ov.A, inv_cov);
    if (!result) {
      throw chol_error("Cholesky decomposition failed in the construction of a gaussian_crf_factor from a moment_gaussian.");
    }
    ov.b = ov.A * mg.mean();
    if (mg.coefficients().size() > 0)
      ov.C = ov.A * mg.coefficients();
  }

  gaussian_crf_factor::gaussian_crf_factor(const canonical_gaussian& cg)
    : base(cg.arguments(), copy_ptr<vector_domain>(new vector_domain())),
      head_(cg.arg_vector()), tail_(), fixed_records_(false),
      conditioned_f(cg), relabeled(false) {
    bool result = chol(ov.A, cg.inf_matrix());
    if (!result) {
      std::cerr << "Could not take Cholesky decomposition of lambda = \n"
                << cg.inf_matrix() << std::endl;
      throw chol_error("Cholesky decomposition failed in the construction of a gaussian_crf_factor from a canonical_gaussian.");
    }
    mat tmpmat;
    result = inv(tmpmat, ov.A * trans(ov.A));
    if (!result) {
      throw inv_error("Matrix inverse failed in the construction of a gaussian_crf_factor from a canonical_gaussian.");
    }
    ov.b = tmpmat * (ov.A * cg.inf_vector());
    // ov.C is empty
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const canonical_gaussian& cg,
                      const vector_domain& Y,
                      const vector_domain& X)
    : base(Y, copy_ptr<vector_domain>(new vector_domain(X))),
      fixed_records_(false), relabeled(false) {

    // Check arguments.
    if (!includes(cg.arguments(), Y) ||
        !includes(cg.arguments(), X) ||
        cg.arguments().size() != Y.size() + X.size()) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::gaussian_crf_factor") +
         "(cg,head_vars,tail_vars,Y,X) given Y,X not matching cg arguments.");
    }

    // Build head_, tail_, relabeling info.
    foreach(vector_variable* v, cg.arg_vector()) {
      if (Y.count(v)) {
        head_.push_back(v);
      } else {
        tail_.push_back(v);
      }
    }

    reset_ov(cg);
  } // gaussian_crf_factor(cg, Y, X)

  gaussian_crf_factor::
  gaussian_crf_factor(const canonical_gaussian& cg,
                      const vector_domain& head_vars,
                      const vector_domain& tail_vars,
                      const vector_domain& Y,
                      const vector_domain& X)
    : base(Y, copy_ptr<vector_domain>(new vector_domain(X))),
      fixed_records_(false) {

    // Check arguments.
    if (!includes(cg.arguments(), Y) ||
        !includes(cg.arguments(), X) ||
        cg.arguments().size() != Y.size() + X.size()) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::gaussian_crf_factor") +
         "(cg,head_vars,tail_vars,Y,X) given Y,X not matching cg arguments.");
    }
    if (cg.arguments().size() != head_vars.size() + tail_vars.size()) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::gaussian_crf_factor") +
         "(cg,head_vars,tail_vars,Y,X) given head_vars,tail_vars" +
         " not matching cg arguments in size.");
    }

    // Build head_, tail_, relabeling info.
    foreach(vector_variable* v, cg.arg_vector()) {
      if (head_vars.count(v)) {
        head_.push_back(v);
        if (X.count(v))
          X_in_head_.push_back(v);
      } else if (tail_vars.count(v)) {
        tail_.push_back(v);
        if (X.count(v))
          X_in_tail_.push_back(v);
      } else {
        throw std::invalid_argument
          (std::string("gaussian_crf_factor::gaussian_crf_factor") +
           "(cg,head_vars,tail_vars,Y,X) given head_vars,tail_vars" +
           " not matching cg arguments.");
      }
    }
    if (head_vars == Y) {
      relabeled = false;
      X_in_tail_.clear();
    } else {
      relabeled = true;
    }

    reset_ov(cg);
  } // gaussian_crf_factor(cg, head_vars, tail_vars, Y, X)

  gaussian_crf_factor::gaussian_crf_factor(double c)
    : base(), fixed_records_(false), conditioned_f(c), relabeled(false) { }

  gaussian_crf_factor::
  gaussian_crf_factor(const vector_domain& out_args,
                      const vector_domain& in_args,
                      const vector_var_vector& head_,
                      const vector_var_vector& tail_,
                      const mat& A, const vec& b, const mat& C)
    : base(out_args, in_args), head_(head_), tail_(tail_), ov(A,b,C),
      fixed_records_(false), relabeled(false) {
    if (out_args != make_domain(head_) || in_args != make_domain(tail_)) {
      relabel_outputs_inputs(out_args, in_args);
    }
  }

  void gaussian_crf_factor::save(oarchive & ar) const {
    base::save(ar);
    bool const_value = (head_.size() == 0);
    ar << const_value;
    if (const_value) {
      ar << conditioned_f;
    } else {
      ar << head_ << tail_ << ov << relabeled;
      if (relabeled)
        ar << X_in_head_ << X_in_tail_;
      // TO DO: AFTER NIPS, REMOVE THIS RELABELING STUFF SINCE IT IS NOT NEEDED.
    }
  }

  void gaussian_crf_factor::load(iarchive & ar) {
    base::load(ar);
    bool const_value;
    ar >> const_value;
    if (const_value) {
      ar >> conditioned_f;
      relabeled = false;
    } else {
      ar >> head_ >> tail_ >> ov >> relabeled;
      // TO DO: AFTER NIPS, REMOVE LOADING THIS RELABELING STUFF SINCE IT IS NOT NEEDED.
      if (relabeled) {
        ar >> X_in_head_ >> X_in_tail_;
      }
      vector_domain tmpY(output_arguments());
      vector_domain tmpX(input_arguments());
      relabel_outputs_inputs(tmpY, tmpX);
    }
    fixed_records_ = false;
  }

  // Public methods: Getters
  // =========================================================================

  const vector_var_vector& gaussian_crf_factor::head() const { return head_; }

  const vector_var_vector& gaussian_crf_factor::tail() const { return tail_; }

  void gaussian_crf_factor::
  print(std::ostream& out, bool print_Y, bool print_X, bool print_vals) const {
    out << "F[";
    if (print_Y)
      out << Ydomain_;
    else
      out << "*";
    out << ", ";
    if (print_X)
      out << (*Xdomain_ptr_);
    else
      out << "*";
    out << "]\n";
    if (print_vals) {
      if (relabeled)
        out << "relabeled Y,X: head = " << head_ << ", tail = " << tail_
            << "\n";
      ov.print(out);
    }
  }

  void
  gaussian_crf_factor::relabel_outputs_inputs(const output_domain_type& new_Y,
                                              const input_domain_type& new_X) {
    {
      vector_domain headset(head_.begin(), head_.end());
      vector_domain tailset(tail_.begin(), tail_.end());
      if (includes(new_Y, headset) && includes(new_X, tailset)) {
        relabeled = false;
        return; // No relabeling needed
      }
    }

    Y_in_head_.clear();
    Y_in_tail_.clear();
    X_in_head_.clear();
    X_in_tail_.clear();
    std::vector<size_t> Y_in_head_ov_indices_tmp, Y_in_tail_ov_indices_tmp,
      X_in_head_ov_indices_tmp, X_in_tail_ov_indices_tmp;
    size_t i = 0;
    foreach(vector_variable* v, head_) {
      if (new_Y.count(v)) {
        Y_in_head_.push_back(v);
        assert(new_X.count(v) == 0);
        for (size_t j = 0; j < v->size(); ++j) {
          Y_in_head_ov_indices_tmp.push_back(i);
          ++i;
        }
      } else if (new_X.count(v)) {
        X_in_head_.push_back(v);
        for (size_t j = 0; j < v->size(); ++j) {
          X_in_head_ov_indices_tmp.push_back(i);
          ++i;
        }
      } else {
        assert(false);
      }
    }
    i = 0;
    foreach(vector_variable* v, tail_) {
      if (new_Y.count(v)) {
        Y_in_tail_.push_back(v);
        assert(new_X.count(v) == 0);
        for (size_t j = 0; j < v->size(); ++j) {
          Y_in_tail_ov_indices_tmp.push_back(i);
          ++i;
        }
      } else if (new_X.count(v)) {
        X_in_tail_.push_back(v);
        for (size_t j = 0; j < v->size(); ++j) {
          X_in_tail_ov_indices_tmp.push_back(i);
          ++i;
        }
      } else {
        assert(false);
      }
    }
    Y_in_head_ov_indices_ =
      arma::conv_to<uvec>::from(Y_in_head_ov_indices_tmp);
    Y_in_tail_ov_indices_ =
      arma::conv_to<uvec>::from(Y_in_tail_ov_indices_tmp);
    X_in_head_ov_indices_ =
      arma::conv_to<uvec>::from(X_in_head_ov_indices_tmp);
    X_in_tail_ov_indices_ =
      arma::conv_to<uvec>::from(X_in_tail_ov_indices_tmp);

    Ydomain_.clear();
    Ydomain_.insert(Y_in_head_.begin(), Y_in_head_.end());
    Ydomain_.insert(Y_in_tail_.begin(), Y_in_tail_.end());
    Xdomain_ptr_->clear();
    Xdomain_ptr_->insert(X_in_head_.begin(), X_in_head_.end());
    Xdomain_ptr_->insert(X_in_tail_.begin(), X_in_tail_.end());
    relabeled = true;
  } // relabel_outputs_inputs

  // Public methods: Probabilistic queries
  // =========================================================================

  double gaussian_crf_factor::v(const vector_assignment& a) const {
    return exp(logv(a));
  }

  double gaussian_crf_factor::v(const record_type& r) const {
    return exp(logv(r));
  }

  double gaussian_crf_factor::logv(const assignment_type& a) const {
    if (head_.size() == 0)
      return conditioned_f(a);
    vec y(zeros<vec>(ov.C.n_rows));
    vec x(zeros<vec>(ov.C.n_cols));
    vector_assignment2vector(a, head_, y);
    vector_assignment2vector(a, tail_, x);
    y = (ov.A * y) - ov.b - (ov.C * x);
    return (-.5) * dot(y, y);
  }

  double gaussian_crf_factor::logv(const record_type& r) const {
    if (head_.size() == 0)
      return conditioned_f(r);
    vec y(zeros<vec>(ov.C.n_rows));
    vec x(zeros<vec>(ov.C.n_cols));
    get_head_tail_values(r, y, x);
    y = (ov.A * y) - ov.b - (ov.C * x);
    return (-.5) * dot(y, y);
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const vector_assignment& a) const {
    if (relabeled) {
      vec C_row_x(zeros<vec>(vector_size(X_in_head_)));
      vec C_col_x(zeros<vec>(vector_size(X_in_tail_)));
      vector_assignment2vector(a, X_in_head_, C_row_x);
      vector_assignment2vector(a, X_in_tail_, C_col_x);
      return condition(C_row_x, C_col_x);
    } else {
      vec x(zeros<vec>(ov.C.n_cols));
      vector_assignment2vector(a, tail_, x);
      return condition(x);
    }
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const record_type& r) const {
    if (relabeled) {
      gaussian_crf_factor gcf(*this);
      gcf.partial_condition(r, output_domain_type(), *Xdomain_ptr_);
      conditioned_f = gcf.get_gaussian<canonical_gaussian>();
      return conditioned_f;
      /*
      vec C_row_x(zeros<vec>(vector_size(X_in_head_)));
      vec C_col_x(zeros<vec>(vector_size(X_in_tail_)));
      get_x_values(r, C_row_x, C_col_x);
      return condition(C_row_x, C_col_x);
      */
    } else {
      vec x(zeros<vec>(ov.C.n_cols));
      get_tail_values(r, x);
      return condition(x);
    }
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const vec& x) const {
    if (relabeled) {
      if (!set_equal(Ydomain_, make_domain(head_)) ||
          !set_equal(*Xdomain_ptr_, make_domain(tail_))) {
        throw std::runtime_error
          (std::string("gaussian_crf_factor::condition(x)") +
           " called on factor with relabeled variables;" +
           " condition(x_in_head, x_in_tail) must be used instead.");
      }
    }
    if (x.size() != ov.C.n_cols) {
      throw std::invalid_argument
        ("gaussian_crf_factor::condition(x) given x of size " +
         to_string(x.size()) + " but expected size " + to_string(ov.C.n_cols));
    }
    if (head_.size() == 0) // If this is a constant factor
      return conditioned_f;
    if (conditioned_f.arg_vector() == head_) { // avoid reallocation
      conditioned_f.inf_matrix() = trans(ov.A) * ov.A;
      if (x.size() == 0) {
        conditioned_f.inf_vector() = trans(ov.A) * ov.b;
      } else {
        conditioned_f.inf_vector() = trans(ov.A) * (ov.b + ov.C * x);
      }
      conditioned_f.log_multiplier() = 0;
    } else {
      if (x.size() == 0) {
        conditioned_f.reset(head_, trans(ov.A) * ov.A,
                            trans(ov.A) * ov.b);
      } else {
        conditioned_f.reset(head_, trans(ov.A) * ov.A,
                            trans(ov.A) * (ov.b + ov.C * x));
      }
    }
    return conditioned_f;
  } // condition(x)

  const canonical_gaussian&
  gaussian_crf_factor::
  condition(const vec& x_in_head, const vec& x_in_tail) const {
    if (!relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::condition(x_in_head,x_in_tail)") +
         " called on factor whose vars had not been relabeled.");
    }
    if (head_.size() == 0) // If this is a constant factor
      return conditioned_f;
    // TO DO: Do this more efficiently.
    moment_gaussian mg(this->get_gaussian<moment_gaussian>());
    vector_assignment a;
    if (x_in_head.size() != vector_size(X_in_head_) ||
        x_in_tail.size() != vector_size(X_in_tail_)) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::condition(x_in_head,x_in_tail)") +
         " given arguments not matching factor variables.");
    }
    add_vector2vector_assignment(X_in_head_, x_in_head, a);
    add_vector2vector_assignment(X_in_tail_, x_in_tail, a);
    conditioned_f = mg.restrict(a);
    return conditioned_f;
  } // condition(x_in_head, x_in_tail)

  gaussian_crf_factor&
  gaussian_crf_factor::
  partial_expectation_in_log_space(const output_domain_type& Y_part) {
    throw std::runtime_error
      (std::string("gaussian_crf_factor::partial_expectation_in_log_space") +
       " NOT YET IMPLEMENTED!");
  } // partial_expectation_in_log_space

  gaussian_crf_factor&
  gaussian_crf_factor::
  partial_expectation_in_log_space(const output_domain_type& Y_part,
                                   const dataset<dense_linear_algebra<> >& ds) {
    assert(!relabeled); // TO DO
    if (ds.size() == 0) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::partial_expectation_in_log_space") +
         " given empty dataset.");
    }

    // TO DO: Make this more efficient.
    canonical_gaussian cg(this->get_gaussian<canonical_gaussian>());
    canonical_gaussian final_cg;

    vector_assignment va;
    mat lambda(zeros(cg.inf_matrix().n_rows, cg.inf_matrix().n_cols));
    vec eta(zeros<vec>(cg.inf_vector().size()));
    foreach(const record_type& r, ds.records()) {
      r.add_to_assignment(Y_part, va);
      canonical_gaussian tmpcg(cg.restrict(va));
      lambda += tmpcg.inf_matrix();
      eta += tmpcg.inf_vector();
      // Ignore log_mult.
      if (final_cg.arguments().size() == 0)
        final_cg = tmpcg;
    }
    lambda /= ds.size();
    eta /= ds.size();
    final_cg.inf_matrix() = lambda;
    final_cg.inf_vector() = eta;

    vector_domain final_head;
    vector_domain final_tail;
    vector_domain tmp_head(head_.begin(), head_.end());
    foreach(vector_variable* v, final_cg.arg_vector()) {
      if (tmp_head.count(v))
        final_head.insert(v);
      else
        final_tail.insert(v);
    }

    gaussian_crf_factor gcf(final_cg, final_head, final_tail,
                            set_difference(this->output_arguments(), Y_part),
                            this->input_arguments());
    this->operator=(gcf);
    return *this;
  } // partial_expectation_in_log_space

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
      (std::string("gaussian_crf_factor::partial_condition (with assignment)")+
       " NOT YET IMPLEMENTED!");
  }

  gaussian_crf_factor&
  gaussian_crf_factor::partial_condition(const record_type& r,
                                         const output_domain_type& Y_part,
                                         const input_domain_type& X_part) {
    // TO DO: Make this efficient.
    vector_domain YX_part(Y_part);
    YX_part.insert(X_part.begin(), X_part.end());

    uvec head_part_indices; // head vars in Y/X_part
    uvec head_retain_indices; // head vars not in Y/X_part
    vector_indices_relative_to_set
      (head_, YX_part, head_part_indices, head_retain_indices);
    uvec tail_part_indices; // tail vars in Y/X_part
    uvec tail_retain_indices; // tail vars not in Y/X_part
    vector_indices_relative_to_set
      (tail_, YX_part, tail_part_indices, tail_retain_indices);

    vec head_part_values;
    vector_var_vector new_head;
    {
      vector_var_vector remove_from_head;
      select_subvector_and_complement
        (head_, YX_part, remove_from_head, new_head);
      r.vector_values(head_part_values, remove_from_head);
    }
    vec tail_part_values;
    vector_var_vector new_tail;
    {
      vector_var_vector remove_from_tail;
      select_subvector_and_complement
        (tail_, YX_part, remove_from_tail, new_tail);
      r.vector_values(tail_part_values, remove_from_tail);
    }

    if (tail_part_values.size() != 0)
      ov.b += columns(ov.C, tail_part_indices) * tail_part_values;
    if (head_part_values.size() != 0)
      ov.b -= columns(ov.A, head_part_indices) * head_part_values;
    ov.A = columns(ov.A, head_retain_indices);
    ov.C = columns(ov.C, tail_retain_indices);

    head_ = new_head;
    tail_ = new_tail;
    fixed_records_ = false; // TO DO: maintain this
    foreach(vector_variable* v, Y_part)
      Ydomain_.erase(v);
    foreach(vector_variable* v, X_part)
      Xdomain_ptr_->erase(v);
    if (relabeled) {
      vector_domain tmpY(Ydomain_);
      vector_domain tmpX(*Xdomain_ptr_);
      relabel_outputs_inputs(tmpY, tmpX);
    }

    return *this;
  } // partial_condition(r, Y_part, X_part)

  double gaussian_crf_factor::log_expected_value(const dataset<dense_linear_algebra<> >& ds) const {
    double val(0.);
    double total_ds_weight(0);
    size_t i(0);
    foreach(const record_type& r, ds.records()) {
      vec y(zeros<vec>(ov.C.n_rows));
      vec x(zeros<vec>(ov.C.n_cols));
      get_head_tail_values(r, y, x);
      y = (ov.A * y) - ov.b - (ov.C * x);
      val += ds.weight(i) * (-.5) * dot(y, y);
      total_ds_weight += ds.weight(i);
      ++i;
    }
    assert(total_ds_weight > 0);
    return (val / total_ds_weight);
  }

  gaussian_crf_factor& gaussian_crf_factor::square_root() {
    ov /= 2;
    return *this;
  }

  gaussian_crf_factor& gaussian_crf_factor::kth_root(double k) {
    assert(k > 0);
    ov /= k;
    return *this;
  }

  // Public: Learning methods from learnable_crf_factor interface
  // =========================================================================

  void gaussian_crf_factor::add_gradient
  (gaussian_crf_factor::optimization_vector& grad,
   const record_type& r, double w) const {
    // grad.A += w (b + Cx - Ay) y'
    // grad.b += -w (b + Cx - Ay)
    // grad.C += -w (b + Cx - Ay) x'
    vec y(zeros<vec>(ov.A.n_rows));
    if (y.size() == 0)
      return;
    vec x(zeros<vec>(ov.C.n_cols));
    get_head_tail_values(r, y, x);
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
                                                  const record_type& r,
                                                  const canonical_gaussian& fy,
                                                  double w) const {
    add_expected_gradient(grad, r, moment_gaussian(fy), w);
  }

  void
  gaussian_crf_factor::add_expected_gradient(optimization_vector& grad,
                                             const record_type& r,
                                             const moment_gaussian& fy,
                                             double w) const {
    if (relabeled) {
      vec mu_x_head(ov.A.n_rows);
      {
        mu_x_head(Y_in_head_ov_indices_) = fy.mean(Y_in_head_);
        vec x_head(X_in_head_ov_indices_.size());
        r.vector_values(x_head, X_in_head_);
        mu_x_head(X_in_head_ov_indices_) = x_head;
      }
      vec mu_x_tail(ov.C.n_cols);
      {
        mu_x_tail(Y_in_tail_ov_indices_) = fy.mean(Y_in_tail_);
        vec x_tail(X_in_tail_ov_indices_.size());
        r.vector_values(x_tail, X_in_tail_);
        mu_x_tail(X_in_tail_ov_indices_) = x_tail;
      }

      vec tmpvec(ov.b);
      if (mu_x_tail.size() != 0)
        tmpvec += ov.C * mu_x_tail;
      if (mu_x_head.size() != 0)
        tmpvec -= ov.A * mu_x_head;
      tmpvec *= w;

      if (Y_in_head_.size() != 0) {
        set_submatrix
          (grad.A, span(0,grad.A.n_rows-1), Y_in_head_ov_indices_,
           mat(w * columns(ov.A,Y_in_head_ov_indices_)
               * fy.covariance(Y_in_head_)));
        if (Y_in_tail_.size() != 0)
          set_submatrix
            (grad.A, span(0,grad.A.n_rows-1), Y_in_head_ov_indices_,
             mat(w * columns(ov.C,Y_in_tail_ov_indices_)
                 * fy.covariance(Y_in_tail_, Y_in_head_)));
      }
      if (mu_x_head.size() != 0)
        grad.A += outer_product(tmpvec, mu_x_head);

      grad.b -= tmpvec;

      if (Y_in_tail_.size() != 0) {
        subtract_submatrix
          (grad.C,
           span(0,grad.A.n_rows-1),
           Y_in_tail_ov_indices_,
           mat(w * columns(ov.C,Y_in_tail_ov_indices_)
               * fy.covariance(Y_in_tail_)));
        if (Y_in_head_.size() != 0)
          add_submatrix
            (grad.C,
             span(0,grad.A.n_rows-1),
             Y_in_head_ov_indices_,
             mat(w * columns(ov.A,Y_in_head_ov_indices_)
                 * fy.covariance(Y_in_head_,Y_in_tail_)));
      }
      if (mu_x_tail.size() != 0)
        grad.C -= outer_product(tmpvec, mu_x_tail);

    } else { // else not relabeled

      // grad.A += w [(b + Cx) mu' - A (Sigma + mu mu')]
      // grad.b += -w [b + Cx - A mu]
      // grad.C += -w [(b + Cx - A mu) x']
      vec mu(fy.mean(head_));
      if (mu.size() == 0)
        return;
      vec x(zeros<vec>(ov.C.n_cols));
      get_tail_values(r, x);
      vec tmpvec(ov.b);
      if (x.size() != 0)
        tmpvec += ov.C * x;

      grad.A += tmpvec * trans(w * mu);
      grad.A -= w * ov.A * (fy.covariance(head_) + mu * trans(mu));
      tmpvec -= ov.A * mu;
      tmpvec *= w;
      grad.b -= tmpvec;
      if (x.size() != 0)
        grad.C -= outer_product(tmpvec, x);
    }
  } // add_expected_gradient(grad, r, fy, w);

  void
  gaussian_crf_factor::
  add_combined_gradient(optimization_vector& grad, const record_type& r,
                        const canonical_gaussian& fy, double w) const {
    add_combined_gradient(grad, r, moment_gaussian(fy), w);
  }

  void
  gaussian_crf_factor::add_combined_gradient
  (optimization_vector& grad, const record_type& r,
   const moment_gaussian& fy, double w) const {
    if (relabeled) {
      add_gradient(grad, r, w);
      add_expected_gradient(grad, r, fy, -1 * w);
    } else {
      vec y(zeros<vec>(ov.A.n_rows));
      if (y.size() == 0)
        return;
      vec x(zeros<vec>(ov.C.n_cols));
      get_head_tail_values(r, y, x);
      vec mu(fy.mean(head_));

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
      grad.A += w * ov.A * fy.covariance(head_);
      grad.b -= tmpvec2;
      if (x.size() != 0)
        grad.C -= (tmpvec + tmpvec2) * trans(x);
    }
  }

  void gaussian_crf_factor::
  add_hessian_diag(optimization_vector& hessian, const record_type& r,
                   double w) const {
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_hessian_diag") +
         " not yet fully implemented.");
    }
    vec tmpvec(ov.A.n_rows);
    if (tmpvec.size() != 0) {
      get_head_values(r, tmpvec);
      tmpvec %= tmpvec;
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.A.n_cols; ++j)
        hessian.A.col(j) = hessian.A.col(j) - tmpvec[j];
    }
    hessian.b -= w;
    if (ov.C.n_cols != 0) {
      tmpvec.set_size(ov.C.n_cols);
      get_tail_values(r, tmpvec);
      tmpvec %= tmpvec;
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.C.n_cols; ++j)
        hessian.C.col(j) = hessian.C.col(j) - tmpvec[j];
    }
  }

  void gaussian_crf_factor::
  add_expected_hessian_diag(optimization_vector& hessian,
                            const record_type& r,
                            const canonical_gaussian& fy, double w) const {
    add_expected_hessian_diag(hessian, r, moment_gaussian(fy), w);
  }

  void gaussian_crf_factor::
  add_expected_hessian_diag(optimization_vector& hessian,
                            const record_type& r,
                            const moment_gaussian& fy, double w) const {
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_expected_hessian_diag") +
         " not yet fully implemented.");
    }
    vec tmpvec(fy.mean(head_));
    if (tmpvec.size() != 0) {
      tmpvec %= tmpvec;
      tmpvec += fy.covariance_diag(head_);
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.A.n_cols; ++j)
        hessian.A.col(j) = hessian.A.col(j) - tmpvec[j];
    }
    hessian.b -= w;
    if (ov.C.n_cols != 0) {
      tmpvec.set_size(ov.C.n_cols);
      get_tail_values(r, tmpvec);
      tmpvec %= tmpvec;
      if (w != 1)
        tmpvec *= w;
      for (size_t j(0); j < ov.C.n_cols; ++j)
        hessian.C.col(j) = hessian.C.col(j) - tmpvec[j];
    }
  }

  void gaussian_crf_factor::
  add_expected_squared_gradient(optimization_vector& sqrgrad,
                                const record_type& r,
                                const canonical_gaussian& fy, double w) const{
    add_expected_squared_gradient(sqrgrad, r, moment_gaussian(fy), w);
  }

  void gaussian_crf_factor::
  add_expected_squared_gradient(optimization_vector& sqrgrad,
                                const record_type& r,
                                const moment_gaussian& fy, double w) const {
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_expected_squared_gradient") +
         " not yet fully implemented.");
    }
    vec mu(fy.mean(head_));
    if (mu.size() == 0)
      return;
    mat cov(fy.covariance(head_));
    vec x(zeros<vec>(ov.C.n_cols));
    get_tail_values(r, x);
    mat tmpmat(outer_product(mu, mu));
    tmpmat += cov;
    vec Gdiag(tmpmat.diag());   // Gdiag(j) = G_{jj}
    tmpmat *= trans(ov.A);  // (tmpmat = GA' now)
    vec b_Cx(ov.b);
    if (x.size() != 0)
      b_Cx += ov.C * x;        // b_Cx(i) = b_i + C_{i.} \cdot x
    vec A_mu(ov.A * mu);       // A_mu(i) = A_{i.} \cdot \mu
    vec tmpvec(b_Cx);
    tmpvec *= -2.;
    tmpvec += A_mu;
    mat tmpmat2(outer_product(tmpvec, mu));
    tmpmat2 += trans(tmpmat);
    tmpmat2 *= (2. * w);
    mat A_sigma(ov.A);
    A_sigma *= cov;            // A_sigma(i,j) = A_{i.} \cdot \sigma_{j.}
    tmpmat2 %= A_sigma;
    sqrgrad.A += tmpmat2;
    tmpvec = A_mu;
    tmpvec *= -2.;
    tmpvec += b_Cx;
    tmpvec %= b_Cx;
    tmpmat %= trans(ov.A);
    tmpvec += sum(tmpmat, 1);
    if (w != 1)
      tmpvec *= w;
    sqrgrad.A += outer_product(tmpvec, Gdiag);
    sqrgrad.b += tmpvec;
    if (x.size() != 0) {
      x %= x;
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
            reg.lambdas[0] * (dot(ov.C, ov.C) + dot(ov.b, ov.b));
        if (reg.lambdas[1] != 0)
          val -= reg.lambdas[1] * dot(ov.A, ov.A);
        return (.5 * val);
      }
    case 3: // L2 on b,C and log_det((A'A)^-1)
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -= .5 * reg.lambdas[0]
            * (dot(ov.C, ov.C) + dot(ov.b, ov.b));
//        val += (reg.lambdas[1] + 2 * ov.A.n_rows) * log_det(ov.A);
        val += (.5 * reg.lambdas[1] + ov.A.n_rows) * log_det(mat(trans(ov.A) * ov.A));
        return val;
      }
    case 4: // L2 on b,C and [ log_det((A'A)^-1) + tr(A'A) ]
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -= .5 * reg.lambdas[0]
            * (dot(ov.C, ov.C) + dot(ov.b,ov.b));
//        val += (reg.lambdas[1] + 2 * ov.A.n_rows) * log_det(ov.A);
        val += (.5 * reg.lambdas[1] + ov.A.n_rows) * log_det(mat(trans(ov.A) * ov.A));
        if (reg.lambdas[1] != 0)
          val -= .5 * reg.lambdas[1] * dot(ov.A,ov.A);
        return val;
      }
    case 5: // L2 on b,C and tr((A'A)^-1)
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -=
            reg.lambdas[0] * (dot(ov.C, ov.C) + dot(ov.b,ov.b));
        if (reg.lambdas[1] != 0) {
          vec eig_AtA;
          bool result = eig_sym(eig_AtA, trans(ov.A) * ov.A);
          if (!result) {
            throw std::runtime_error("eig_sym failed in gaussian_crf_factor::regularization_penalty().");
          }
          val -= accu(reg.lambdas[1] / eig_AtA);
        }
        return (.5 * val);
      }
    case 6: // L2 on b,C and [ -log_det((A'A)^-1) + tr((A'A)^-1) ]
      {
        double val(0.);
        if (reg.lambdas[0] != 0)
          val -= .5 * reg.lambdas[0]
            * (dot(ov.C, ov.C) + dot(ov.b,ov.b));
        if (reg.lambdas[1] != 0) {
//          val -= reg.lambdas[1] * log_det(ov.A);
          val -= (.5 * reg.lambdas[1]) * log_det(mat(trans(ov.A) * ov.A));
          vec eig_AtA;
          bool result = eig_sym(eig_AtA, trans(ov.A) * ov.A);
          if (!result) {
            throw std::runtime_error("eig_sym failed in gaussian_crf_factor::regularization_penalty().");
          }
          val -= .5 * accu(reg.lambdas[1] / eig_AtA);
        }
        return val;
      }
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  } // regularization_penalty

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
    case 3: // L2 on b,C and log_det((A'A)^-1)
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      {
        mat AtA_inv_At;
        bool result = solve(AtA_inv_At, trans(ov.A) * ov.A, trans(ov.A));
        if (!result) {
          throw ls_solve_chol_error("solve failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A -=
          w * (reg.lambdas[1] + 2 * ov.A.n_rows) * trans(AtA_inv_At);
      }
      return;
    case 4: // L2 on b,C and [ log_det((A'A)^-1) + tr(A'A) ]
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      {
        mat AtA_inv;
        bool result = inv(AtA_inv, trans(ov.A) * ov.A);
        if (!result) {
          throw inv_error("Matrix inverse failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A -=
          ov.A * w * ((reg.lambdas[1] + 2 * ov.A.n_rows) * AtA_inv
                      + reg.lambdas[1] * eye(AtA_inv.n_rows,AtA_inv.n_rows));
      }
      return;
    case 5: // L2 on b,C and tr((A'A)^-1)
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      if (reg.lambdas[1] != 0) {
        mat tmpmat(trans(ov.A));
        tmpmat *= ov.A;
        bool result = solve(tmpmat, tmpmat * tmpmat, trans(ov.A));
        if (!result) {
          throw ls_solve_chol_error("solve failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A += w * reg.lambdas[1] * trans(tmpmat);
      }
      return;
    case 6: // L2 on b,C and [ -log_det((A'A)^-1) + tr((A'A)^-1) ]
      if (reg.lambdas[0] != 0) {
        grad.C -= w * reg.lambdas[0] * ov.C;
        grad.b -= w * reg.lambdas[0] * ov.b;
      }
      if (reg.lambdas[1] != 0) {
        mat AtA_inv;
        bool result = inv(AtA_inv, trans(ov.A) * ov.A);
        if (!result) {
          throw inv_error("Matrix inverse failed in gaussian_crf_factor::add_regularization_gradient().");
        }
        grad.A += w * reg.lambdas[1]
          * ov.A * AtA_inv * (AtA_inv - eye(ov.A.n_rows, ov.A.n_rows));
      }
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  } // add_regularization_gradient

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
    case 3: // L2 on b,C and log_det((A'A)^-1)
    case 4: // L2 on b,C and [ log_det((A'A)^-1) + tr(A'A) ]
    case 5: // L2 on b,C and tr((A'A)^-1)
    case 6: // L2 on b,C and [ -log_det((A'A)^-1) + tr((A'A)^-1) ]
      std::cerr << "GAUSSIAN_CRF_FACTOR::ADD_REGULARIZATION_HESSIAN_DIAG()"
                << " NOT YET FULLY IMPLEMENTED!!" << std::endl;
      assert(false);
    default:
      throw std::invalid_argument
        (std::string("table_crf_factor::add_regularization_hessian_diag()") +
         " given bad regularization argument.");
    }
  } // add_regularization_hessian_diag

  // Public methods: Operators
  // =========================================================================

  gaussian_crf_factor&
  gaussian_crf_factor::operator*=(const gaussian_crf_factor& other) {
    if (!set_disjoint(this->output_arguments(), other.input_arguments()) ||
        !set_disjoint(this->input_arguments(), other.output_arguments())) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::operator*=") +
         " tried to multiply two factors with at least one variable common" +
         " to one factor's Y and the other factor's X.");
    }
    // TO DO: Implement this more efficiently, and maintain fixed_record_ if
    //        it was set for both this and other.
    canonical_gaussian this_mg(this->get_gaussian<canonical_gaussian>());
    canonical_gaussian other_mg(other.get_gaussian<canonical_gaussian>());
    this_mg *= other_mg;
    gaussian_crf_factor tmp_gcf(this_mg);
    if (relabeled || other.relabeled) {
      tmp_gcf.relabel_outputs_inputs(set_union(this->output_arguments(),
                                               other.output_arguments()),
                                     set_union(this->input_arguments(),
                                               other.input_arguments()));
    }
    this->operator=(tmp_gcf);
    return *this;
  }

  // Templated methods from gaussian_crf_factor
  //============================================================================

  template <>
  moment_gaussian
  gaussian_crf_factor::get_gaussian<moment_gaussian>() const {
    if (head_.size() == 0)
      return moment_gaussian(conditioned_f);
    mat sigma;
    bool result =
      solve(sigma, trans(ov.A) * ov.A, eye(ov.A.n_rows,ov.A.n_rows));
    if (!result) {
      throw std::runtime_error
        ("Cholesky decomposition failed in gaussian_crf_factor::get_gaussian");
    }
    mat sigma_At(sigma * trans(ov.A));
    mat mg_coeff;
    if (ov.C.size() > 0)
      mg_coeff = sigma_At * ov.C;
    return moment_gaussian(head_, sigma_At * ov.b, sigma, tail_, mg_coeff);
  }

  template <>
  canonical_gaussian
  gaussian_crf_factor::get_gaussian<canonical_gaussian>() const {
    if (head_.size() > 0) {
      if (tail_.size() > 0) {
        vector_var_vector YX(sill::concat(head_, tail_));
        size_t YXsize = vector_size(YX);
        mat AtA(trans(ov.A) * ov.A);
        vec btA(trans(ov.A) * ov.b);
        mat AtC(trans(ov.A) * ov.C);
        mat AtA_inv_AtC;
        if (!solve(AtA_inv_AtC, AtA, AtC)) {
          throw std::runtime_error
            (std::string("gaussian_crf_factor::") +
             "get_gaussian<canonical_gaussian>: Cholesky decomposition failed");
        }
        vec eta(YXsize);
        eta.subvec(span(0,ov.A.n_rows-1)) = btA;
        eta.subvec(span(ov.A.n_rows, YXsize-1)) = trans(AtA_inv_AtC) * btA;
        mat lambda(YXsize, YXsize);
        lambda(span(0,ov.A.n_rows-1), span(0,ov.A.n_rows-1)) = AtA;
        lambda(span(0,ov.A.n_rows-1), span(ov.A.n_rows, YXsize-1)) = - AtC;
        lambda(span(ov.A.n_rows, YXsize-1), span(0,ov.A.n_rows-1)) =
          - trans(AtC);
        lambda(span(ov.A.n_rows, YXsize-1), span(ov.A.n_rows, YXsize-1)) =
          trans(AtC) * AtA_inv_AtC;
        return canonical_gaussian(YX, lambda, eta);
      } else {
        return canonical_gaussian(head_, trans(ov.A) * ov.A,
                                  trans(ov.A) * ov.b);
      }
    } else {
      if (tail_.size() == 0)
        return canonical_gaussian();
      return canonical_gaussian(tail_, trans(ov.C) * ov.C, trans(ov.C) * ov.b);
    }
  }

}  // namespace sill

#include <sill/macros_undef.hpp>
