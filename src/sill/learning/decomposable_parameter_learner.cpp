#include <sill/learning/decomposable_parameter_learner.hpp>

namespace sill {

  bool decomposable_parameter_learner_parameters::valid(bool verbose) const {
    if (regularization < 0) {
      if (verbose)
        std::cerr << "decomposable_parameter_learner_parameters::valid() found"
                  << " negative regularization: " << regularization
                  << std::endl;
      return false;
    }
    return true;
  } // decomposable_parameter_learner_parameters::valid

}; // namespace sill
