
#include <prl/learning/crf/crf_parameter_learner.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  bool crf_parameter_learner_parameters::valid(bool print_warnings) const {
    // TO DO: print_warnings
    if (!gm_params.valid())
      return false;
    if (perturb < 0)
      return false;
    return true;
  } // crf_parameter_learner_parameters::valid

}  // namespace prl

#include <prl/macros_undef.hpp>
