
#include <sill/factor/operations.hpp>
#include <sill/factor/table_crf_factor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Public methods: Constructors, getters, helpers
  // =========================================================================

  void
  table_crf_factor::relabel_outputs_inputs(const output_domain_type& new_Y,
                                           const input_domain_type& new_X) {
    if (!valid_output_input_relabeling(output_arguments(), input_arguments(),
                                       new_Y,new_X)) {
      throw std::invalid_argument
        (std::string("table_crf_factor::relabel_outputs_inputs") +
         " given new_Y,new_X whose union did not equal the union of the" +
         " old Y,X.");
    }
    Ydomain_ = new_Y;
    Xdomain_ptr_->operator=(new_X);
    optimize_variable_order();
  }

  // Public methods: Probabilistic queries
  // =========================================================================

  double table_crf_factor::v(const finite_assignment& a) const {
    if (log_space_)
      return std::exp(f.f(a));
    else
      return f.f(a);
  }

  double table_crf_factor::v(const finite_record& r) const {
    if (log_space_)
      return std::exp(f.f(r.finite_assignment()));
    else
      return f.f(r.finite_assignment());
  }

  const table_factor&
  table_crf_factor::condition(const finite_assignment& a) const {
    f.f.restrict(conditioned_f, a, input_arguments(), true);
    if (log_space_)
      conditioned_f.update(exponent<double>());
    return conditioned_f;
  }

  const table_factor&
  table_crf_factor::condition(const finite_record& r) const {
    f.f.restrict_aligned(r, restrict_map, conditioned_f);
    if (log_space_)
      conditioned_f.update(exponent<double>());
    return conditioned_f;
  }

  table_crf_factor&
  table_crf_factor::
  partial_expectation_in_log_space(const output_domain_type& Y_part) {
    bool was_in_log_space = log_space();
    if (!was_in_log_space)
      convert_to_log_space();
    size_t num_removed_Y_assignments =
      num_assignments(set_intersect(Y_part, Ydomain_));
    this->marginalize_out(Y_part);
    f.f /= num_removed_Y_assignments;
    if (!was_in_log_space)
      convert_to_real_space();
    optimize_variable_order();
    return *this;
  }

  table_crf_factor&
  table_crf_factor::
  partial_expectation_in_log_space(const output_domain_type& Y_part,
                                   const dataset<la_type>& ds) {
    if (!set_disjoint(Y_part, *Xdomain_ptr_)) {
      throw std::invalid_argument("table_crf_factor::partial_expectation_in_log_space(Y_part, ds) given Y_part which overlaps with the factor's input variables X.");
    }
    bool was_in_log_space = log_space();
    if (!was_in_log_space)
      convert_to_log_space();
    table_factor new_f;
    table_factor tmp_f;
    foreach(const finite_record& r, ds.records()) {
      f.f.restrict(tmp_f, r, Y_part);
      tmp_f /= ds.size();
      new_f += tmp_f;
    }
    f.f = new_f;
    foreach(finite_variable* v, Y_part)
      Ydomain_.erase(v);
    if (!was_in_log_space)
      convert_to_real_space();
    optimize_variable_order();
    return *this;
  }

  table_crf_factor&
  table_crf_factor::marginalize_out(const output_domain_type& Y_other) {
    table_factor new_f;
    if (!set_disjoint(Y_other, *Xdomain_ptr_)) {
      throw std::invalid_argument("table_crf_factor::marginalize_out(Y_other) given Y_other which overlaps with the factor's input variables X.");
    }
    foreach(finite_variable* v, Y_other)
      Ydomain_.erase(v);
    f.f.marginal(new_f, set_union(Ydomain_, *Xdomain_ptr_));
    f.f = new_f;
    optimize_variable_order();
    return *this;
  }

  table_crf_factor&
  table_crf_factor::partial_condition(const finite_assignment& a,
                                      const finite_domain& Y_part,
                                      const finite_domain& X_part) {
    table_factor new_f;
    f.f.restrict(new_f, a, set_union(Y_part, X_part), true);
    f.f = new_f;
    foreach(finite_variable* v, Y_part)
      Ydomain_.erase(v);
    foreach(finite_variable* v, X_part)
      Xdomain_ptr_->erase(v);
    optimize_variable_order();
    return *this;
  }

  table_crf_factor&
  table_crf_factor::partial_condition(const finite_record& r,
                                      const finite_domain& Y_part,
                                      const finite_domain& X_part) {
    table_factor new_f;
    f.f.restrict(new_f, r, set_union(Y_part, X_part), true);
    f.f = new_f;
    foreach(finite_variable* v, Y_part)
      Ydomain_.erase(v);
    foreach(finite_variable* v, X_part)
      Xdomain_ptr_->erase(v);
    optimize_variable_order();
    return *this;
  }

  double table_crf_factor::log_expected_value(const dataset<la_type>& ds) const {
    double val(0);
    output_factor_type tmp_fctr;
    double total_ds_weight(0);
    size_t i(0);
    if (log_space()) {
      foreach(const finite_record& r, ds.records()) {
        f.f.restrict(tmp_fctr, r, input_arguments(), true);
        val += ds.weight(i) * tmp_fctr(r);
        total_ds_weight += ds.weight(i);
        ++i;
      }
    } else {
      foreach(const finite_record& r, ds.records()) {
        f.f.restrict(tmp_fctr, r, input_arguments(), true);
        val += ds.weight(i) * std::log(tmp_fctr(r));
        total_ds_weight += ds.weight(i);
        ++i;
      }
    }
    assert(total_ds_weight > 0);
    return (val / total_ds_weight);        
  }

  table_crf_factor&
  table_crf_factor::combine_in(const table_crf_factor& other, op_type op) {
    if (arguments().size() > 0) {
      throw std::runtime_error
        ("table_crf_factor::combine_in NOT YET FULLY IMPLEMENTED!");
    }
    switch (op) {
    case no_op:
      break;
    case sum_op:
    case minus_op:
    case product_op:
      throw std::runtime_error
        ("table_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
      break;
    case divides_op:
      {
        double myval = this->v(finite_assignment());
        this->operator=(other);
        f.reciprocal();
        f *= myval;
      }
      break;
    case max_op:
    case min_op:
    case and_op:
    case or_op:
      throw std::runtime_error
        ("table_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
      break;
    default:
      assert(false);
    }
    optimize_variable_order();
    return *this;
  }

  table_crf_factor&
  table_crf_factor::combine_in(const constant_factor& other, op_type op) {
    throw std::runtime_error
      ("table_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
  }

  table_crf_factor& table_crf_factor::square_root() {
    if (log_space_)
      f /= 2;
    else
      f.elem_square_root();
    return *this;
  }

  // Public: Learning methods from learnable_crf_factor interface
  // =========================================================================

  void table_crf_factor::add_gradient(table_factor_opt_vector& grad,
                                      const finite_record& r, double w) const {
    if (log_space_) {
      grad.f(r) += w;
    } else {
      double val(f.f(r));
      if (val != 0)
        grad.f(r) += w / val;
      else
        grad.f(r) += w * std::numeric_limits<double>::infinity();
    }
  }

  void table_crf_factor::add_expected_gradient
  (table_factor_opt_vector& grad, const finite_record& r,
   const table_factor& fy, double w) const {
    assert(set_equal(Ydomain_, fy.arguments()));
    finite_assignment fa(r.assignment(input_arguments()));
    if (log_space_) {
      foreach(const finite_assignment& fa2, assignments(Ydomain_)) {
        finite_assignment::const_iterator fa2_end(fa2.end());
        for (finite_assignment::const_iterator fa2_it(fa2.begin());
             fa2_it != fa2_end; ++fa2_it)
          fa[fa2_it->first] = fa2_it->second;
        grad.f(fa) += w * fy(fa2);
      }
    } else {
      foreach(const finite_assignment& fa2, assignments(Ydomain_)) {
        finite_assignment::const_iterator fa2_end(fa2.end());
        for (finite_assignment::const_iterator fa2_it(fa2.begin());
             fa2_it != fa2_end; ++fa2_it)
          fa[fa2_it->first] = fa2_it->second;
        double val(f.f(r));
        if (val != 0)
          grad.f(fa) += w * fy(fa2) / val;
        else
          grad.f(fa) += w * std::numeric_limits<double>::infinity();
      }
    }
  }

  void
  table_crf_factor::add_combined_gradient
  (optimization_vector& grad, const finite_record& r,
   const output_factor_type& fy, double w) const {
    add_gradient(grad, r, w);
    add_expected_gradient(grad, r, fy, -1. * w);
  }

  void
  table_crf_factor::add_hessian_diag
  (optimization_vector& hessian, const finite_record& r, double w) const {
    if (!log_space_) {
      assert(false); // TO DO
    }
  }

  void
  table_crf_factor::add_expected_hessian_diag(optimization_vector& hessian,
                                              const finite_record& r,
                                              const table_factor& fy,
                                              double w) const {
    if (!log_space_) {
      assert(false); // TO DO
    }
  }

  void
  table_crf_factor::add_expected_squared_gradient(optimization_vector& sqrgrad,
                                                  const finite_record& r,
                                                  const table_factor& fy,
                                                  double w) const {
    add_expected_gradient(sqrgrad, r, fy, w);
  }

  void table_crf_factor::add_regularization_hessian_diag
  (optimization_vector& hd, const regularization_type& reg, double w) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return;
    case 2:
      if (log_space_) {
        if (reg.lambdas[0] != 0)
          hd -= w * reg.lambdas[0];
      } else
        assert(false);
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  double table_crf_factor::
  regularization_penalty(const regularization_type& reg) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return 0.;
    case 2:
      if (reg.lambdas[0] == 0)
        return 0.;
      else {
        if (log_space_)
          return (-.5 * reg.lambdas[0] * f.inner_prod(f));
        else
          assert(false); // TO DO
      }
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  void table_crf_factor::
  add_regularization_gradient(table_factor_opt_vector& grad,
                              const regularization_type& reg, double w) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return;
    case 2:
      if (log_space_) {
        if (reg.lambdas[0] != 0)
          grad -= f * w * reg.lambdas[0];
      } else
        assert(false);
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  // Public methods: Operators
  // =========================================================================

  table_crf_factor&
  table_crf_factor::operator*=(const table_crf_factor& other) {
    if (!set_disjoint(this->output_arguments(), other.input_arguments()) ||
        !set_disjoint(this->input_arguments(), other.output_arguments())) {
      throw std::runtime_error("table_crf_factor::operator*= tried to multiply two factors with at least one variable common to one factor's Y and the other factor's X.");
    }
    if (log_space_) {
      if (other.log_space()) {
        f.f += other.f.f;
      } else {
        // TO DO: This copy could be avoided.
        table_factor other_f_f(other.f.f);
        other_f_f.update(logarithm<double>());
        f.f += other_f_f;
      }
    } else {
      if (other.log_space()) {
        convert_to_log_space();
        f.f += other.f.f;
        convert_to_real_space();
      } else {
        f.f *= other.f.f;
      }
    }
    Ydomain_.insert(other.output_arguments().begin(),
                    other.output_arguments().end());
    Xdomain_ptr_->insert(other.input_arguments().begin(),
                         other.input_arguments().end());
    optimize_variable_order();
    return *this;
  }

  // Private methods
  // =========================================================================

  void table_crf_factor::optimize_variable_order() {
    // Check if Y precedes X in f's argument sequence.
    bool good_order = true;
    if (output_arguments().size() != f.f.arguments().size()) {
      good_order = false;
    } else {
      for (size_t i = 0; i < output_arguments().size(); ++i) {
        if (!output_arguments().count(f.f.arg_list()[i])) {
          good_order = false;
          break;
        }
      }
    }
    if (!good_order) {
      output_var_vector_type Y_vec(output_arguments().begin(),
                                   output_arguments().end());
      var_vector_type YX_vec(Y_vec);
      YX_vec.insert(YX_vec.end(),
                    input_arguments().begin(), input_arguments().end());
      table_factor_opt_vector new_f(YX_vec, 0);
      foreach(const finite_assignment& fa, f.f.assignments())
        new_f.f(fa) = f.f(fa);
      f = new_f;
      conditioned_f = table_factor(Y_vec, 0);
    } else {
      // Make sure conditioned_f's arguments are in the same order as f.f.
      if (conditioned_f.arguments().size() != output_arguments().size()) {
        good_order = false;
      } else {
        for (size_t i = 0; i < output_arguments().size(); ++i) {
          if (f.f.arg_list()[i] != conditioned_f.arg_list()[i]) {
            good_order = false;
            break;
          }
        }
      }
      if (!good_order) {
        output_var_vector_type Y_vec(output_arguments().size(), NULL);
        for (size_t i = 0; i < output_arguments().size(); ++i)
          Y_vec[i] = f.f.arg_list()[i];
        conditioned_f = table_factor(Y_vec, 0);
      }
    }
    if (restrict_map.size() != f.f.arguments().size())
      restrict_map = table_factor::shape_type(f.f.arguments().size(), 0);
  } // optimize_variable_order

}  // namespace sill

#include <sill/macros_undef.hpp>
