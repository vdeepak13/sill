
#include <prl/factor/random.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  table_factor
  make_associative_factor(finite_variable* Yi, finite_variable* Yj, double s) {
    assert(Yi && Yj);
    assert(Yi->size() == Yj->size());
    table_factor f(make_domain<finite_variable>(Yi,Yj), 1.);
    finite_assignment fa;
    for (size_t k(0); k < Yi->size(); ++k) {
      fa[Yi] = k;
      fa[Yj] = k;
      f(fa) = s;
    }
    return f;
  }

} // namespace prl

#include <prl/macros_undef.hpp>
