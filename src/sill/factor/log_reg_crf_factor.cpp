
#include <sill/factor/log_reg_crf_factor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Public methods: Constructors, getters, helpers
  // =========================================================================

  log_reg_crf_factor::
  log_reg_crf_factor(const output_domain_type& Y_,
                     const input_domain_type& X_)
    : base(Y_, copy_ptr<domain>(new domain(X_.begin(), X_.end()))),
      conditioned_f(Y_, 0.) {
    assert(false); // I NEED TO INITIALIZE mlr_ptr.
  }

  log_reg_crf_factor::
  log_reg_crf_factor(boost::shared_ptr<multiclass2multilabel> mlr_ptr,
                     double smoothing, const finite_domain& Y_,
                     copy_ptr<domain> X_ptr_)
    : base(Y_, X_ptr_), mlr_ptr(mlr_ptr), smoothing(smoothing), 
      conditioned_f(Y_, 0.) {
    assert(mlr_ptr.get() != NULL);
    mlr_ptr->prepare_record_for_base(tmp_record);
  }

  void log_reg_crf_factor::print(std::ostream& out) const {
    base::print(out);
    if (mlr_ptr)
      out << *mlr_ptr;
  }

  // Public methods: Probabilistic queries
  // =========================================================================

  const table_factor&
  log_reg_crf_factor::condition(const assignment& a) const {
    conditioned_f = mlr_ptr->probabilities(a);
    conditioned_f += smoothing;
    conditioned_f.normalize();
    return conditioned_f;
    // TO DO: MAKE THE ABOVE MORE EFFICIENT IF NECESSARY (I.E., MAKE USE OF
    //        THE PRE-ALLOCATED conditioned_f.
  }

  const table_factor&
  log_reg_crf_factor::condition(const record_type& r) const {
    conditioned_f = mlr_ptr->probabilities(r);
    conditioned_f += smoothing;
    conditioned_f.normalize();
    return conditioned_f;
    // TO DO: MAKE THE ABOVE MORE EFFICIENT IF NECESSARY (I.E., MAKE USE OF
    //        THE PRE-ALLOCATED conditioned_f.
  }

  /*
  double log_reg_crf_factor::log_expected_value(const dataset& ds) const {
    double val(0);
    table_factor tmp_fctr;
    double total_ds_weight(0);
    size_t i(0);
    foreach(const assignment& a, ds.assignments()) {
      assignment tmpa(a);
      foreach(finite_variable* v, output_arguments())
        tmpa.finite().erase(v);
      tmp_fctr = condition(tmpa);
      val += ds.weight(i) * std::log(tmp_fctr(a.finite()));
      total_ds_weight += ds.weight(i);
      ++i;
    }
    assert(total_ds_weight > 0);
    return (val / total_ds_weight);        
  }
  */

  // Public: Learning-related methods from crf_factor interface
  // =========================================================================

  const multiclass_logistic_regression<>::opt_variables&
  log_reg_crf_factor::weights() const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<>*>(base_ptr.get());
    return base_mlr_ptr->weights();
  }

  multiclass_logistic_regression<>::opt_variables&
  log_reg_crf_factor::weights() {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    multiclass_logistic_regression<>* base_mlr_ptr
      = dynamic_cast<multiclass_logistic_regression<>*>(base_ptr.get());
    return base_mlr_ptr->weights();
  }

  // Public: Learning methods from learnable_crf_factor interface
  // =========================================================================

  void
  log_reg_crf_factor::add_gradient
  (multiclass_logistic_regression<>::opt_variables& grad, const record_type& r,
   double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_gradient(grad, tmp_record, w);
  }

  void
  log_reg_crf_factor::add_expected_gradient
  (multiclass_logistic_regression<>::opt_variables& grad,
   const record_type& r, const table_factor& fy,
   double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_expected_gradient(grad, tmp_record, fy, w);
  }

  void
  log_reg_crf_factor::add_combined_gradient
  (optimization_vector& grad, const record_type& r,
   const output_factor_type& fy, double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_combined_gradient(grad, tmp_record, fy, w);
  }

  void
  log_reg_crf_factor::add_hessian_diag
  (optimization_vector& hessian, const record_type& r, double w) const {
    return; // This is 0.
  }

  void
  log_reg_crf_factor::add_expected_hessian_diag
  (optimization_vector& hessian, const record_type& r,
   const table_factor& fy, double w) const {
    return; // This is 0.
  }

  void
  log_reg_crf_factor::add_expected_squared_gradient
  (optimization_vector& sqrgrad, const record_type& r,
   const table_factor& fy, double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_expected_squared_gradient(sqrgrad, tmp_record, fy, w);
  }

  double log_reg_crf_factor::regularization_penalty
  (const regularization_type& reg) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return 0.;
    case 2:
      if (reg.lambdas[0] == 0) {
        return 0.;
      } else {
        const optimization_vector& tmpov = weights();
        return (-.5 * reg.lambdas[0] * tmpov.dot(tmpov));
      }
    default:
      throw std::invalid_argument("log_reg_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  void log_reg_crf_factor::add_regularization_gradient
  (optimization_vector& grad, const regularization_type& reg, double w) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return;
    case 2:
      {
        const optimization_vector& tmpov = weights();
        if (reg.lambdas[0] != 0)
          grad -= tmpov * w * reg.lambdas[0];
      }
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  void log_reg_crf_factor::add_regularization_hessian_diag
  (optimization_vector& hd, const regularization_type& reg, double w) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return;
    case 2:
      if (reg.lambdas[0] != 0)
        hd -= w * reg.lambdas[0];
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

}  // namespace sill

#include <sill/macros_undef.hpp>
