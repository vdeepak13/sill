
#include <sill/learning/crf/crf_parameter_learner.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  bool crf_parameter_learner_parameters::valid(bool print_warnings) const {
    // TO DO: print_warnings
    if (!gm_params.valid())
      return false;
    if (perturb < 0)
      return false;
    return true;
  } // crf_parameter_learner_parameters::valid

}  // namespace sill

#include <sill/macros_undef.hpp>
