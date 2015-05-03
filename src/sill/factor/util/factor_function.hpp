#ifndef SILL_FACTOR_FUNCTION_HPP
#define SILL_FACTOR_FUNCTION_HPP

#include <functional>

namespace sill {

  template <typename F>
  using marginal_fn = std::function<F(const typename F::domain_type&)>;

  template <typename F>
  using conditional_fn = std::function<F(const typename F::domain_type&,
                                         const typename F::domain_type&)>;
} // namespace sill

#endif
