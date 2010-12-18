#include <sill/learning/discriminative/free_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  void record2vector(vec& vals, const vector_var_vector& vars,
                     const vector_record& r) {
    size_t i(0); // index into vals
    foreach(vector_variable* v, vars) {
      for (size_t j(0); j < v->size(); ++j) {
        vals[i] = r.vector(v, j);
        ++i;
      }
    }
  }

  void assignment2vector(vec& vals, const vector_var_vector& vars,
                         const vector_assignment& r) {
    size_t i(0); // index into vals
    foreach(vector_variable* v, vars) {
      const vec& val = safe_get(r, v);
      for (size_t j(0); j < v->size(); ++j) {
        vals[i] = val[j];
        ++i;
      }
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>
