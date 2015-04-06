#ifndef SILL_FINITE_ASSIGNMENT_HPP
#define SILL_FINITE_ASSIGNMENT_HPP

#include <sill/argument/domain.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/datastructure/finite_index.hpp>

#include <unordered_map>

namespace sill {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to finite variables.
   * Each finite variable is mapped to a finite value.
   */
  typedef std::unordered_map<finite_variable*, size_t> finite_assignment;

  /**
   * Returns the number of variables for which both finite_assignments
   * agree.
   */
  inline size_t
  agreement(const finite_assignment& a1, const finite_assignment& a2) {
    const finite_assignment& a = a1.size() < a2.size() ? a1 : a2;
    const finite_assignment& b = a1.size() < a2.size() ? a2 : a1;
    size_t count = 0;
    for (const auto& p : a) {
      auto it = b.find(p.first);
      count += (it != b.end()) && (it->second == p.second);
    }
    return count;
  }

  /**
   * Returns the finite values in the assignment for a subset of arguments
   * in the order specified by the given domain.
   * \relates finite_assignment
   */
  inline finite_index
  extract(const finite_assignment& a,
          const domain<finite_variable*>& dom) {
    finite_index result;
    result.reserve(dom.size());
    for (finite_variable* v : dom) {
      result.push_back(a.at(v));
    }
    return result;
  }

  //! @}

} // namespace sill

#endif
