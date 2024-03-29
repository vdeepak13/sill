#ifndef SILL_ARGUMENTS_FUNCTOR_HPP
#define SILL_ARGUMENTS_FUNCTOR_HPP

#include <functional>

#include <sill/factor/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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

  /**
   * A functor that returns the output arguments of a CRF factor.
   * \ingroup factor_types
   */
  template <typename F>
  struct output_arguments_functor 
    : public std::unary_function<const F&,
                                 const typename F::output_domain_type&> {
//    concept_assert((CRFfactor<F>));
    const typename F::output_domain_type& operator()(const F& f) const {
      return f.output_arguments();
    }
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
