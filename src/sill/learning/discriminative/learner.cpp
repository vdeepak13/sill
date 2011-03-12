
#include <sill/learning/discriminative/learner.hpp>

namespace sill {

  // Free functions
  //==========================================================================

  std::ostream& operator<<(std::ostream& out, const learner& l) {
    l.print(out);
    return out;
  }

} // namespace sill
