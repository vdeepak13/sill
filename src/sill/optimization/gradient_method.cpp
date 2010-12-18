
#include <sill/optimization/gradient_method.hpp>

namespace sill {

  bool gradient_method_parameters::valid(bool print_warnings) const {
    switch (step_type) {
    case SINGLE_OPT_STEP:
      if (!single_opt_step_params.valid())
        return false;
      break;
    case LINE_SEARCH:
    case LINE_SEARCH_WITH_GRAD:
      if (!ls_params.valid())
        return false;
      break;
    default:
      assert(false);
    }
    if (convergence_zero < 0)
      return false;
    return true;
  }

}; // namespace sill
