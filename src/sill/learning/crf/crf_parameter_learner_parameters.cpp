
#include <sill/learning/crf/crf_parameter_learner_parameters.hpp>

namespace sill {

  crf_parameter_learner_parameters::crf_parameter_learner_parameters()
    : opt_method(real_optimizer_builder::CONJUGATE_GRADIENT),
      regularization(2), lambdas(1,0), init_iterations(0),
      init_time_limit(0), perturb(0), random_seed(time(NULL)),
      no_shared_computation(false), keep_fixed_records(false), debug(0) { }

  bool crf_parameter_learner_parameters::valid(bool print_warnings) const {
    // TO DO: print_warnings
    if (!gm_params.valid())
      return false;
    if (perturb < 0)
      return false;
    return true;
  } // crf_parameter_learner_parameters::valid

}  // namespace sill
