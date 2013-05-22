
#include <sill/math/linear_algebra/armadillo.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  uvec sequence(size_t from, size_t to) {
    assert(from <= to);
    uvec s(to - from);
    size_t i = 0;
    while (from < to) {
      s[i] = from;
      ++from;
      ++i;
    }
    return s;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
