#ifndef SILL_PAIRWISE_COMPATIBLE_HPP
#define SILL_PAIRWISE_COMPATIBLE_HPP

#include <type_traits>

namespace sill {

  /**
   * Represents the true_type if the two factor types have the same
   * real_type, result_type, variable_type, and assignment_type.
   */
  template <typename F, typename G>
  struct pairwise_compatible : public std::integral_constant<
    bool,
    std::is_same<typename F::real_type, typename G::real_type>::value &&
    std::is_same<typename F::result_type, typename G::result_type>::value &&
    std::is_same<typename F::variable_type, typename G::variable_type>::value &&
    std::is_same<typename F::assignment_type, typename G::assignment_type>::value
  > { };

} // namespace sill

#endif