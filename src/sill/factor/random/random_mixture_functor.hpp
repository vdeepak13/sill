#ifndef SILL_RANDOM_MIXTURE_FUNCTOR_HPP
#define SILL_RANDOM_MIXTURE_FUNCTOR_HPP

#include <sill/factor/mixture.hpp>
#include <sill/factor/random/random_factor_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random mixture factors.
   */
  template <typename F>
  struct random_mixture_functor
    : random_factor_functor<mixture<F> > {

    typedef random_factor_functor<mixture<F> > base;

    typedef typename base::variable_type variable_type;
    typedef typename base::domain_type   domain_type;

    // Parameters
    //==========================================================================

    //! Number of components to include in the mixture.
    //!  (default = 3)
    size_t k;

    // Public methods
    //==========================================================================

    //! Constructor.
    explicit random_mixture_functor(random_factor_functor<F>& subfactor_functor)
      : k(3), subfactor_functor(subfactor_functor) { }

    using base::generate_marginal;
    using base::generate_conditional;

    /**
     * Generate a marginal factor P(X) using the stored parameters.
     */
    mixture<F> generate_marginal(const domain_type& X) {
      mixture<F> mix(k, F(X));
      for (size_t i = 0; i < k; ++i)
        mix[i] = subfactor_functor.generate_marginal(X);
      return mix;
    } // generate_marginal

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    mixture<F>
    generate_conditional(const domain_type& Y, const domain_type& X) {
      mixture<F> mix(k, F(set_union(Y,X)));
      for (size_t i = 0; i < k; ++i)
        mix[i] = subfactor_functor.generate_conditional(Y, X);
      return mix;
    } // generate_conditional

    //! Generate a variable of the appropriate type and dimensionality,
    //! using the given name.
    variable_type*
    generate_variable(universe& u, const std::string& name = "") const {
      return subfactor_functor.generate_variable(u, name);
    }

    // Private data and methods
    //==========================================================================
  private:

    random_factor_functor<F>& subfactor_functor;

    random_mixture_functor();

    random_mixture_functor(const random_mixture_functor& other);

  }; // struct random_mixture_functor

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_MIXTURE_FUNCTOR_HPP
