
#ifndef SILL_ERROR_MEASURES_HPP
#define SILL_ERROR_MEASURES_HPP

#include <sill/base/assignment.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Namespace with methods for computing measures of error.
   * This is useful for, e.g., implementing decomposable::per_label_accuracy()
   * so that it handles both finite and vector variables.
   */
  namespace error_measures {

    /**
     * Squared error(a,b):
     *  - finite variables: for each i in vars, +1 if a_i != b_i; else 0
     *  - vector variables: sum_{i in vars} (a_i - b_i)^2
     */
    template <typename AssignmentType>
    double squared_error
    (const AssignmentType& a, const AssignmentType& b,
     const std::set<typename AssignmentType::key_type>& vars);

    //! Template specialization; see original definition.
    template <>
    double squared_error<finite_assignment>
    (const finite_assignment& a, const finite_assignment& b,
     const finite_domain& vars);

    //! Template specialization; see original definition.
    template <>
    double squared_error<vector_assignment>
    (const vector_assignment& a, const vector_assignment& b,
     const vector_domain& vars);

  } // namespace error_measures

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_ERROR_MEASURES_HPP
