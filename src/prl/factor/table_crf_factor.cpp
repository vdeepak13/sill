
#include <sill/factor/operations.hpp>
#include <sill/factor/table_crf_factor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Public methods: Constructors, getters, helpers
  // =========================================================================

  void
  table_crf_factor::relabel_outputs_inputs(const output_domain_type& new_Y,
                                           const input_domain_type& new_X) {
    if (new_Y.size() + new_X.size() !=
        output_arguments().size() + input_arguments().size()) {
      throw std::invalid_argument("table_crf_factor::relabel_outputs_inputs given new_Y,new_X whose size did not match the old Y,X.");
    }
    domain_type args(arguments());
    foreach(output_variable_type* v, new_Y)
      args.erase(v);
    foreach(input_variable_type* v, new_X)
      args.erase(v);
    if (args.size() != 0) {
      throw std::invalid_argument("table_crf_factor::relabel_outputs_inputs given new_Y,new_X whose union did not equal the union of the old Y,X.");
    }
    Ydomain_ = new_Y;
    Xdomain_ptr_->operator=(new_X);
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
    f.f.restrict(conditioned_f, r, input_arguments(), true);
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
    return *this;
  }

  table_crf_factor&
  table_crf_factor::
  partial_expectation_in_log_space(const output_domain_type& Y_part,
                                   const dataset& ds) {
    if (!set_disjoint(Y_part, *Xdomain_ptr_)) {
      throw std::invalid_argument("table_crf_factor::partial_expectation_in_log_space(Y_part, ds) given Y_part which overlaps with the factor's input variables X.");
    }
    bool was_in_log_space = log_space();
    if (!was_in_log_space)
      convert_to_log_space();
    table_factor new_f;
    table_factor tmp_f;
    foreach(const record& r, ds.records()) {
      f.f.restrict(tmp_f, r, Y_part);
      tmp_f /= ds.size();
      new_f += tmp_f;
    }
    f.f = new_f;
    foreach(finite_variable* v, Y_part)
      Ydomain_.erase(v);
    if (!was_in_log_space)
      convert_to_real_space();
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
    return *this;
  }

  double table_crf_factor::log_expected_value(const dataset& ds) const {
    double val(0);
    output_factor_type tmp_fctr;
    double total_ds_weight(0);
    size_t i(0);
    if (log_space()) {
      foreach(const record& r, ds.records()) {
        f.f.restrict(tmp_fctr, r, input_arguments(), true);
        val += ds.weight(i) * tmp_fctr(r);
        total_ds_weight += ds.weight(i);
        ++i;
      }
    } else {
      foreach(const record& r, ds.records()) {
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
    return *this;
  }

  table_crf_factor&
  table_crf_factor::combine_in(const constant_factor& other, op_type op) {
    throw std::runtime_error
      ("table_crf_factor::combine_in NOT FULLY IMPLEMENTED!");
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
    finite_assignment fa(r.finite_assignment());
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

}  // namespace sill

#include <sill/macros_undef.hpp>
