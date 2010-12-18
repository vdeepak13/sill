#include <sill/math/free_functions.hpp>

namespace sill {

  double round(double value) {
    return ceil(value - .5);
  }

} // namespace sill
