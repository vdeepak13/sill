#include <sstream>
#include <boost/lexical_cast.hpp>

#include <prl/math/vector.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/macros_def.hpp>

namespace prl {
  

  // Free functions
  //============================================================================
  vec concat(const forward_range<const vec&> vectors) {
    // compute the size of the resulting vector
    size_t n = 0;
    foreach(const vec& v, vectors) n += v.size();
    vec result(n);

    // assign the vectors to the right indices
    n = 0;
    foreach(const vec& v, vectors) {
      result.set_subvector(irange(n, n+v.size()), v);
      n += v.size();
    }
    return result;
  }

} // namespace prl

