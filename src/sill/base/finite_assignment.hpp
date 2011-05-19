#ifndef SILL_FINITE_ASSIGNMENT_HPP
#define SILL_FINITE_ASSIGNMENT_HPP

#include <sill/base/finite_variable.hpp>
#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup base_types
  //! @{

  //! An assignment to a set of finite variables
  typedef std::map<finite_variable*, size_t> finite_assignment;

  //! Returns a new assignment for the given variable-value pair
  inline finite_assignment make_assignment(finite_variable* var, size_t value) {
    finite_assignment a;
    a.insert(std::make_pair(var, value));
    return a;
  }

  //! Returns the number of variable values for which both finite_assignments
  //! agree.
  inline size_t assignment_agreement(const finite_assignment &fa1, 
                                     const finite_assignment &fa2) {
    finite_domain allvars = set_intersect(keys(fa1), keys(fa2));
    size_t count = 0;
    foreach(finite_variable *v, allvars) {
      count += (safe_get(fa1, v) == safe_get(fa2, v));
    }
    return count;
  }

  //! @}

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
