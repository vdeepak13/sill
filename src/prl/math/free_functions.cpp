#include <prl/math/free_functions.hpp>

namespace prl {

  double round(double value) {
    return ceil(value - .5);
  }

} // namespace prl
