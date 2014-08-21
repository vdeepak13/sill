#include <sill/learning/crf/crf_parameter_learner_parameters.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  crf_parameter_learner_parameters::crf_parameter_learner_parameters()
    : regularization(2), lambdas(zeros<vec>(1)), init_iterations(10000),
      init_time_limit(0), learning_objective(MLE), perturb(0),
      random_seed(time(NULL)), keep_fixed_records(false), debug(0),
      no_shared_computation(false),
      opt_method(real_optimizer_builder::CONJUGATE_GRADIENT),
      cg_update_method(0), lbfgs_M(10) { }

  void crf_parameter_learner_parameters::check() const {
    assert(regularization == 0 || regularization == 2);
    foreach(double val, lambdas)
      assert(val >= 0);
    assert(learning_objective <= MPLE);
    assert(perturb >= 0);
    assert(opt_method <= real_optimizer_builder::STOCHASTIC_GRADIENT);
    assert(gm_params.valid());
    assert(cg_update_method == 0);
    assert(lbfgs_M != 0);
  }

  void crf_parameter_learner_parameters::save(oarchive& ar) const {
    ar << regularization << lambdas << init_iterations << init_time_limit
       << learning_objective << perturb << random_seed << keep_fixed_records
       << debug << no_shared_computation << opt_method << gm_params
       << cg_update_method << lbfgs_M;
  }

  void crf_parameter_learner_parameters::load(iarchive& ar) {
    ar >> regularization >> lambdas >> init_iterations >> init_time_limit
       >> learning_objective >> perturb >> random_seed >> keep_fixed_records
       >> debug >> no_shared_computation >> opt_method >> gm_params
       >> cg_update_method >> lbfgs_M;
  }

  oarchive&
  operator<<(oarchive& a,
             crf_parameter_learner_parameters::learning_objective_enum val) {
    a << (size_t)(val);
    return a;
  }

  iarchive&
  operator>>(iarchive& a,
             crf_parameter_learner_parameters::learning_objective_enum& val) {
    size_t tmp;
    a >> tmp;
    val = (crf_parameter_learner_parameters::learning_objective_enum)(tmp);
    return a;
  }

}  // namespace sill

#include <sill/macros_undef.hpp>
