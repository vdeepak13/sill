#ifndef SILL_RANDOM_CANONICAL_GAUSSIAN_FUNCTOR_HPP
#define SILL_RANDOM_CANONICAL_GAUSSIAN_FUNCTOR_HPP

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/random/random_moment_gaussian_functor.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random canonical_gaussian factors.
   * This struct is a wrapper for random_moment_gaussian_functor.
   */
  struct random_canonical_gaussian_functor
    : random_factor_functor_i<canonical_gaussian> {

    typedef random_factor_functor_i<canonical_gaussian> base;

    typedef vector_variable variable_type;
    typedef vector_domain   domain_type;

    // Public data
    //==========================================================================

    random_moment_gaussian_functor rmg_func;

    // Public methods
    //==========================================================================

    random_canonical_gaussian_functor() { }

    //! Constructor.
    explicit
    random_canonical_gaussian_functor
    (const random_moment_gaussian_functor& rmg_func);

    using base::generate_marginal;
    using base::generate_conditional;

    /**
     * Generate a marginal factor P(X) using the stored parameters.
     */
    canonical_gaussian generate_marginal(const domain_type& X);

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    canonical_gaussian
    generate_conditional(const domain_type& Y, const domain_type& X);

    //! Generate a variable of the appropriate type and dimensionality,
    //! using the given name.
    variable_type*
    generate_variable(universe& u, const std::string& name = "") const;

    //! Set random seed.
    void seed(unsigned random_seed = time(NULL));

  }; // struct random_canonical_gaussian_functor

  //! @} group factor_random

} // namespace sill

#endif // SILL_RANDOM_CANONICAL_GAUSSIAN_FUNCTOR_HPP
