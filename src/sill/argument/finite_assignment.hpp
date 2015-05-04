#ifndef SILL_FINITE_ASSIGNMENT_HPP
#define SILL_FINITE_ASSIGNMENT_HPP

#include <sill/argument/basic_domain.hpp>
#include <sill/argument/variable.hpp>
#include <sill/datastructure/finite_index.hpp>

#include <unordered_map>

namespace sill {

  //! \addtogroup argument_types
  //! @{

  /**
   * A type that represents an assignment to finite variables.
   * Each variable is mapped to a finite value.
   *
   * \tparam Var a type representing variables, such as sill::variable
   */
  template <typename Var = variable>
  using finite_assignment = std::unordered_map<Var, size_t>;

  /**
   * Returns the number of variables for which both finite_assignments
   * agree.
   * \relates finite_assignment
   */
  template <typename Var>
  size_t agreement(const finite_assignment<Var>& a1,
                   const finite_assignment<Var>& a2) {
    const finite_assignment<Var>& a = a1.size() < a2.size() ? a1 : a2;
    const finite_assignment<Var>& b = a1.size() < a2.size() ? a2 : a1;
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
  template <typename Var>
  finite_index extract(const finite_assignment<Var>& a,
                       const basic_domain<Var>& dom,
                       size_t start = 0) {
    assert(start <= dom.size());
    finite_index result(dom.size() - start);
    for (size_t i = start; i < dom.size(); ++i) {
      result[i - start] = a.at(dom[i]);
    }
    return result;
  }

  //! @}

} // namespace sill

#endif
