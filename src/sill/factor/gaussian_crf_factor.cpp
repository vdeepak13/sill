
#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/operations.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/record_conversions.hpp>
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
      fixed_records_(false), conditioned_f(Y_, 0.), relabeled(false) {
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
      fixed_records_(false), conditioned_f(Y_, 0.), relabeled(false) {
    ov.zeros();
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const optimization_vector& ov,
                      const vector_var_vector& Y_,
                      const vector_var_vector& X_)
    : base(make_domain(Y_),
           copy_ptr<vector_domain>(new vector_domain(make_domain(X_)))),
      head_(Y_), tail_(X_), ov(ov), fixed_records_(false),
      conditioned_f(Y_, 0.), relabeled(false) {
    if (!ov.valid_size())
      throw std::invalid_argument
        (std::string("gaussian_crf_factor constructor:") +
         " ov dimensions do not match each other.");
    if ((ov.A.size1() != Y_.size()) || (ov.C.size2() != X_.size()))
      throw std::invalid_argument
        ("gaussian_crf_factor constructor: ov dimensions do not match Y,X.");
  }

  gaussian_crf_factor::
  gaussian_crf_factor(const linear_regression& lr, const dataset& ds)
    : base(make_domain(lr.Yvector()),
           copy_ptr<vector_domain>
           (new vector_domain(make_domain(lr.Xvector())))),
      head_(lr.Yvector()), tail_(lr.Xvector()),
      fixed_records_(false), conditioned_f(head_, 0.), relabeled(false) {
    assert(head_.size() > 0);
    mat ds_cov;
    if (vector_size(head_) > 1) {
      ds.covariance(ds_cov, head_);
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
      head_(mg.head()), tail_(mg.tail()), fixed_records_(false),
      conditioned_f(head_, 0.), relabeled(false) {
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
      head_(cg.argument_list()), tail_(), fixed_records_(false),
      conditioned_f(cg), relabeled(false) {
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
    foreach(vector_variable* v, cg.argument_list()) {
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

    // Build ov.
    if (head_.size() > 0) {
      if (tail_.size() > 0) {
        ivec head_ind; // indices in cg for head
        cg.indices(head_, head_ind);
        ivec tail_ind; // indices in cg for tail
        cg.indices(tail_, tail_ind);
        bool result = chol(cg.inf_matrix()(head_ind, head_ind), ov.A);
        if (!result) {
          std::cerr << "Could not take Cholesky decomposition of lambda = \n"
                    << cg.inf_matrix()(head_ind, head_ind) << std::endl;
          throw chol_error
            (std::string("gaussian_crf_factor::gaussian_crf_factor") +
             "(cg,head_vars,tail_vars,Y,X): Cholesky decomposition failed.");
        }
        mat AAt_inv;
        result = inv(ov.A * ov.A.transpose(), AAt_inv);
        if (!result) {
          throw inv_error
            (std::string("gaussian_crf_factor::gaussian_crf_factor") +
             "(cg,head_vars,tail_vars,Y,X): Matrix inverse failed.");
        }
        ov.b = AAt_inv * (ov.A * cg.inf_vector()(head_ind));
        ov.C = AAt_inv * (ov.A * (- cg.inf_matrix()(head_ind, tail_ind)));
      } else {
        bool result = chol(cg.inf_matrix(), ov.A);
        if (!result) {
          std::cerr << "Could not take Cholesky decomposition of lambda = \n"
                    << cg.inf_matrix() << std::endl;
          throw chol_error
            (std::string("gaussian_crf_factor::gaussian_crf_factor") +
             "(cg,head_vars,tail_vars,Y,X): Cholesky decomposition failed.");
        }
        mat AAt_inv;
        result = inv(ov.A * ov.A.transpose(), AAt_inv);
        if (!result) {
          throw inv_error
            (std::string("gaussian_crf_factor::gaussian_crf_factor") +
             "(cg,head_vars,tail_vars,Y,X): Matrix inverse failed.");
        }
        ov.b = AAt_inv * (ov.A * cg.inf_vector());
        // ov.C is empty
      }
    } else {
      if (tail_.size() > 0) {
        // ov.A is empty
        bool result = chol(cg.inf_matrix(), ov.C);
        if (!result) {
          std::cerr << "Could not take Cholesky decomposition of lambda = \n"
                    << cg.inf_matrix() << std::endl;
          throw chol_error
            (std::string("gaussian_crf_factor::gaussian_crf_factor") +
             "(cg,head_vars,tail_vars,Y,X): Cholesky decomposition failed.");
        }
        mat CCt_inv;
        result = inv(ov.C * ov.C.transpose(), CCt_inv);
        if (!result) {
          throw inv_error
            (std::string("gaussian_crf_factor::gaussian_crf_factor") +
             "(cg,head_vars,tail_vars,Y,X): Matrix inverse failed.");
        }
        ov.b = CCt_inv * (ov.C * cg.inf_vector());
      } else {
        // No arguments.
        conditioned_f = cg;
      }
    }
  } // gaussian_crf_factor(cg, head_vars, tail_vars, Y, X)

  gaussian_crf_factor::gaussian_crf_factor(const constant_factor& cf)
    : base(), fixed_records_(false), conditioned_f(cf), relabeled(false) { }

  gaussian_crf_factor::gaussian_crf_factor(double c)
    : base(), fixed_records_(false), conditioned_f(c), relabeled(false) { }

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
      if (relabeled)
        ar >> X_in_head_ >> X_in_tail_;
    }
    fixed_records_ = false;
  }

  // Public methods: Getters
  // =========================================================================

  const vector_var_vector& gaussian_crf_factor::head() const {
    // RIGHT HERE NOW: CHECK EVERYWHERE THIS IS CALLED TO HANDLE 'relabeled'.
    return head_;
  }

  const vector_var_vector& gaussian_crf_factor::tail() const {
    // RIGHT HERE NOW: CHECK EVERYWHERE THIS IS CALLED TO HANDLE 'relabeled'.
    return tail_;
  }

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
    if (!valid_output_input_relabeling(output_arguments(), input_arguments(),
                                       new_Y, new_X)) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::relabel_outputs_inputs") +
         " given new_Y,new_X whose union did not equal the union of the" +
         " old Y,X.");
    }
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::relabel_outputs_inputs") +
         " not yet fully implemented!");
    } else {
      X_in_head_.clear();
      X_in_tail_.clear();
      foreach(vector_variable* v, new_X) {
        if (Ydomain_.count(v) != 0)
          X_in_head_.push_back(v);
        else
          X_in_tail_.push_back(v);
      }
    }
    Ydomain_ = new_Y;
    Xdomain_ptr_->operator=(new_X);
    relabeled = true;
  } // relabel_outputs_inputs

  // Public methods: Probabilistic queries
  // =========================================================================

  double gaussian_crf_factor::v(const vector_assignment& a) const {
    return exp(logv(a));
  }

  double gaussian_crf_factor::v(const vector_record& r) const {
    return exp(logv(r));
  }

  double gaussian_crf_factor::logv(const assignment_type& a) const {
    if (head_.size() == 0)
      return conditioned_f(a);
    vec y(ov.C.size1(), 0);
    vec x(ov.C.size2(), 0);
    vector_assignment2vector(a, head_, y);
    vector_assignment2vector(a, tail_, x);
    y = (ov.A * y) - ov.b - (ov.C * x);
    return (-.5) * inner_prod(y, y);
  }

  double gaussian_crf_factor::logv(const record_type& r) const {
    if (head_.size() == 0)
      return conditioned_f(r);
    vec y(ov.C.size1(), 0);
    vec x(ov.C.size2(), 0);
    get_yx_values(r, y, x);
    y = (ov.A * y) - ov.b - (ov.C * x);
    return (-.5) * inner_prod(y, y);
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const vector_assignment& a) const {
    if (relabeled) {
      vec C_row_x(X_in_head_.size(), 0);
      vec C_col_x(X_in_tail_.size(), 0);
      vector_assignment2vector(a, X_in_head_, C_row_x);
      vector_assignment2vector(a, X_in_tail_, C_col_x);
      return condition(C_row_x, C_col_x);
    } else {
      vec x(ov.C.size2(), 0);
      vector_assignment2vector(a, tail_, x);
      return condition(x);
    }
  }

  const canonical_gaussian&
  gaussian_crf_factor::condition(const vector_record& r) const {
    if (relabeled) {
      gaussian_crf_factor gcf(*this);
      gcf.partial_condition(r, output_domain_type(), *Xdomain_ptr_);
      conditioned_f = gcf.get_gaussian<canonical_gaussian>();
      return conditioned_f;
      /*
      vec C_row_x(X_in_head_.size(), 0);
      vec C_col_x(X_in_tail_.size(), 0);
      get_x_values(r, C_row_x, C_col_x);
      return condition(C_row_x, C_col_x);
      */
    } else {
      vec x(ov.C.size2(), 0);
      get_x_values(r, x);
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
    if (x.size() != ov.C.size2()) {
      throw std::invalid_argument
        ("gaussian_crf_factor::condition(x) given x of size " +
         to_string(x.size()) + " but expected size " + to_string(ov.C.size2()));
    }
    if (head_.size() == 0) // If this is a constant factor
      return conditioned_f;
    if (conditioned_f.argument_list() == head_) { // avoid reallocation
      conditioned_f.inf_matrix() = ov.A.transpose() * ov.A;
      if (x.size() == 0) {
        conditioned_f.inf_vector() = ov.A.transpose() * ov.b;
      } else {
        conditioned_f.inf_vector() = ov.A.transpose() * (ov.b + ov.C * x);
      }
      conditioned_f.log_multiplier() = 0;
    } else {
      if (x.size() == 0) {
        conditioned_f.reset(head_, ov.A.transpose() * ov.A,
                            ov.A.transpose() * ov.b);
      } else {
        conditioned_f.reset(head_, ov.A.transpose() * ov.A,
                            ov.A.transpose() * (ov.b + ov.C * x));
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
    if (x_in_head.size() != X_in_head_.size() ||
        x_in_tail.size() != X_in_tail_.size()) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::condition(x_in_head,x_in_tail)") +
         " given arguments not matching factor variables.");
    }
    add_vector2vector_assignment(X_in_head_, x_in_head, a);
    add_vector2vector_assignment(X_in_tail_, x_in_tail, a);
    mg = mg.restrict(a); // RIGHT HERE NOW: I THINK THERE IS A BUG HERE
    conditioned_f = mg;
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
                                   const dataset& ds) {
    if (ds.size() == 0) {
      throw std::invalid_argument
        (std::string("gaussian_crf_factor::partial_expectation_in_log_space") +
         " given empty dataset.");
    }

    // TO DO: Make this more efficient.
    canonical_gaussian cg(this->get_gaussian<canonical_gaussian>());
    canonical_gaussian final_cg;

    vector_assignment va;
    mat lambda(cg.inf_matrix().size1(), cg.inf_matrix().size2(), 0);
    vec eta(cg.inf_vector().size(), 0);
    foreach(const record& r, ds.records()) {
      r.add_assignment(Y_part, va);
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
    foreach(vector_variable* v, final_cg.argument_list()) {
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

    ivec head_part_indices; // head vars in Y/X_part
    ivec head_retain_indices; // head vars not in Y/X_part
    vector_indices_relative_to_set
      (head_, YX_part, head_part_indices, head_retain_indices);
    ivec tail_part_indices; // tail vars in Y/X_part
    ivec tail_retain_indices; // tail vars not in Y/X_part
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
      ov.b += ov.C.columns(tail_part_indices) * tail_part_values;
    if (head_part_values.size() != 0)
      ov.b -= ov.A.columns(head_part_indices) * head_part_values;
    ov.A = ov.A.columns(head_retain_indices);
    ov.C = ov.C.columns(tail_retain_indices);

    head_ = new_head;
    tail_ = new_tail;
    fixed_records_ = false; // TO DO: maintain this
    if (relabeled) {
      X_in_head_ = select_subvector_complement(X_in_head_, YX_part);
      X_in_tail_ = select_subvector_complement(X_in_tail_, YX_part);
    }
    foreach(vector_variable* v, Y_part)
      Ydomain_.erase(v);
    foreach(vector_variable* v, X_part)
      Xdomain_ptr_->erase(v);

    return *this;
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
        throw std::runtime_error
          ("gaussian_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
        // This is wrong--and it can't even be done in most cases
        // using this representation.
        double myval = this->v(vector_assignment());
        this->operator=(other);
//        ov.reciprocal();
//        ov *= myval;
        ov *= -myval;
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

  gaussian_crf_factor& gaussian_crf_factor::square_root() {
    ov /= 2;
    return *this;
  }

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
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_expected_gradient") +
         " not yet fully implemented.");
    }
    vec mu(fy.mean(head_));
    if (mu.size() == 0)
      return;
    vec x(ov.C.size2(), 0);
    get_x_values(r, x);
    vec tmpvec(ov.b);
    if (x.size() != 0)
      tmpvec += ov.C * x;

    grad.A += outer_product(tmpvec, w * mu);
    grad.A -= w * ov.A * (fy.covariance(head_) + outer_product(mu, mu));
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
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_combined_gradient") +
         " not yet fully implemented.");
    }
    vec y(ov.A.size1(), 0);
    if (y.size() == 0)
      return;
    vec x(ov.C.size2(), 0);
    get_yx_values(r, y, x);
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
      grad.C -= outer_product(tmpvec + tmpvec2, x);
  }

  void gaussian_crf_factor::
  add_hessian_diag(optimization_vector& hessian, const vector_record& r,
                   double w) const {
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_hessian_diag") +
         " not yet fully implemented.");
    }
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
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_expected_hessian_diag") +
         " not yet fully implemented.");
    }
    vec tmpvec(fy.mean(head_));
    if (tmpvec.size() != 0) {
      elem_mult_inplace(tmpvec, tmpvec);
      tmpvec += fy.covariance_diag(head_);
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
    if (relabeled) {
      throw std::runtime_error
        (std::string("gaussian_crf_factor::add_expected_squared_gradient") +
         " not yet fully implemented.");
    }
    vec mu(fy.mean(head_));
    if (mu.size() == 0)
      return;
    mat cov(fy.covariance(head_));
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
    case 3: // L2 on b,C and logdet((A'A)^-1)
    case 4: // L2 on b,C and [ logdet((A'A)^-1) + tr(A'A) ]
    case 5: // L2 on b,C and tr((A'A)^-1)
    case 6: // L2 on b,C and [ -logdet((A'A)^-1) + tr((A'A)^-1) ]
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
    return moment_gaussian(head_, sigma_At * ov.b, sigma, tail_, mg_coeff);
  }

  template <>
  canonical_gaussian
  gaussian_crf_factor::get_gaussian<canonical_gaussian>() const {
    if (head_.size() > 0) {
      if (tail_.size() > 0) {
        vector_var_vector YX(sill::concat(head_, tail_));
        size_t YXsize = vector_size(YX);
        mat AtA(ov.A.transpose() * ov.A);
        vec btA(ov.A.transpose() * ov.b);
        mat AtC(ov.A.transpose() * ov.C);
        mat AtA_inv_AtC;
        if (!ls_solve_chol(AtA, AtC, AtA_inv_AtC)) {
          throw std::runtime_error
            (std::string("gaussian_crf_factor::") +
             "get_gaussian<canonical_gaussian>: Cholesky decomposition failed");
        }
        vec eta(YXsize);
        eta.set_subvector(irange(0,ov.A.size1()), btA);
        eta.set_subvector(irange(ov.A.size1(), YX.size()),
                          AtA_inv_AtC.transpose() * btA);
        mat lambda(YX.size(), YX.size());
        lambda.set_submatrix
          (irange(0,ov.A.size1()), irange(0,ov.A.size1()), AtA);
        lambda.set_submatrix
          (irange(0,ov.A.size1()), irange(ov.A.size1(), YX.size()), - AtC);
        lambda.set_submatrix
          (irange(ov.A.size1(), YX.size()), irange(0,ov.A.size1()),
           - AtC.transpose());
        lambda.set_submatrix
          (irange(ov.A.size1(), YX.size()), irange(ov.A.size1(), YX.size()),
           AtC.transpose() * AtA_inv_AtC);
        return canonical_gaussian(YX, lambda, eta);
      } else {
        return canonical_gaussian(head_, ov.A.transpose() * ov.A,
                                  ov.A.transpose() * ov.b);
      }
    } else {
      if (tail_.size() == 0)
        return canonical_gaussian();
      return canonical_gaussian(tail_, ov.C.transpose() * ov.C,
                                ov.C.transpose() * ov.b);
    }
  }

}  // namespace sill

#include <sill/macros_undef.hpp>
