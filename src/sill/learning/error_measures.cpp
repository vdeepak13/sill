
#include <sill/learning/error_measures.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  namespace error_measures {

    template <>
    double squared_error<finite_assignment>
    (const finite_assignment& a, const finite_assignment& b,
     const finite_domain& vars) {
      double val(0);
      foreach(finite_variable* v, vars)
        if (safe_get(a, v) != safe_get(b, v))
          ++val;
      return val;
    }

    template <>
    double squared_error<vector_assignment>
    (const vector_assignment& a, const vector_assignment& b,
     const vector_domain& vars) {
      double val(0);
      foreach(vector_variable* v, vars) {
        vec diff(safe_get(a, v) - safe_get(b, v));
        val += diff.inner_prod(diff);
      }
      return val;
    }

  } // namespace error_measures

} // namespace sill

#include <sill/macros_undef.hpp>
