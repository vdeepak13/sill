#ifndef SILL_VECTOR_ASSIGNMENT_HPP
#define SILL_VECTOR_ASSIGNMENT_HPP

#include <stdexcept>

#include <sill/base/vector_variable.hpp>
#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup base_types
  //! @{

  //! An assignment to a set of vector variables
  typedef std::map<vector_variable*, vec> vector_assignment;

  //! Returns a new assignment for the given variable-value pair
  inline vector_assignment make_assignment(vector_variable* var,
                                           const vec& value) {
    vector_assignment a;
    a.insert(std::make_pair(var, value));
    return a;
  }

  //! Returns the size of the vector variables in this map.
  //! \relates vector_variable
  size_t vector_size(const vector_assignment& va);

  //! Returns a new assignment for the given (corresponding)
  //! variable vector, value vector pair.
  inline vector_assignment
  make_assignment(const vector_var_vector& vars, const vec& values) {
    vector_assignment a;
    if (vector_size(vars) != values.n_elem)
      throw std::invalid_argument
        ("make_assignment() given vars, values which did not match.");
    size_t k = 0; // index into values
    foreach(vector_variable* var, vars) {
      a[var] = values.subvec(k, k + var->size() - 1);
      k += var->size();
    }
    return a;
  }

  //! @}
}

#endif
