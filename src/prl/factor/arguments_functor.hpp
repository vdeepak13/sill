#ifndef PRL_ARGUMENTS_FUNCTOR_HPP
#define PRL_ARGUMENTS_FUNCTOR_HPP

#include <functional>

#include <prl/factor/concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * A functor that returns the arguments of a factor.
   * \ingroup factor_types
   */
  template <typename F>
  struct arguments_functor 
    : public std::unary_function<const F&, const typename F::domain_type&> {
    concept_assert((Factor<F>));
    const typename F::domain_type& operator()(const F& factor) const {
      return factor.arguments();
    }
  };

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
