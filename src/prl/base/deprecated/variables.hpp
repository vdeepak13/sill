#ifndef PRL_BASE_VARIABLES_HPP
#define PRL_BASE_VARIABLES_HPP

#include <prl/base/variable.hpp>
#include <prl/base/finite_variable.hpp>
#include <prl/base/vector_variable.hpp>
#include <prl/base/finite_assignment_iterator.hpp>

#include <prl/macros_def.hpp>

/**
 * \file variables.hpp
 * \todo The .hpp and .cpp files for variables, domains and assignments
 *       are not organized well.  There should be separate files for things
 *       involving single variables and things involving domains.
 */

namespace prl {

  //! \addtogroup base_types
  //! @{

  //! Number of assignments to the variables in the domain.
  size_t num_assignments(const finite_domain& vars);

  //! Returns a range over all assignments to variables in the domain.
  finite_assignment_range assignments(const finite_domain& vars);

  //! @}

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
