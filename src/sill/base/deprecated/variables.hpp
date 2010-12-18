#ifndef SILL_BASE_VARIABLES_HPP
#define SILL_BASE_VARIABLES_HPP

#include <sill/base/variable.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/finite_assignment_iterator.hpp>

#include <sill/macros_def.hpp>

/**
 * \file variables.hpp
 * \todo The .hpp and .cpp files for variables, domains and assignments
 *       are not organized well.  There should be separate files for things
 *       involving single variables and things involving domains.
 */

namespace sill {

  //! \addtogroup base_types
  //! @{

  //! Number of assignments to the variables in the domain.
  size_t num_assignments(const finite_domain& vars);

  //! Returns a range over all assignments to variables in the domain.
  finite_assignment_range assignments(const finite_domain& vars);

  //! @}

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
