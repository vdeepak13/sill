#ifndef SILL_RANDOM_GAUSSIAN_CRF_FACTOR_FUNCTOR_HPP
#define SILL_RANDOM_GAUSSIAN_CRF_FACTOR_FUNCTOR_HPP

#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/random/random_crf_factor_functor.hpp>
#include <sill/factor/random/random_factor_functor.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random gaussian_crf_factor factors.
   */
  struct random_gaussian_crf_factor_functor
    : random_crf_factor_functor<gaussian_crf_factor> {

    typedef random_crf_factor_functor<gaussian_crf_factor> base;

    // Public methods
    //==========================================================================

    //! Constructor.
    explicit random_gaussian_crf_factor_functor
    (random_factor_functor<moment_gaussian>& mg_factor_func);

    //! Constructor.
    explicit random_gaussian_crf_factor_functor
    (random_factor_functor<canonical_gaussian>& cg_factor_func);

    using base::generate_marginal;
    using base::generate_conditional;

    /**
     * Generate a marginal factor P(Y) using the stored parameters.
     */
    crf_factor_type generate_marginal(const output_domain_type& Y);

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    crf_factor_type generate_conditional(const output_domain_type& Y,
                                         const input_domain_type& X);

    /**
     * Generate an output variable of the appropriate type and dimensionality,
     * using the given name.
     */
    output_variable_type*
    generate_output_variable(universe& u, const std::string& name = "") const;

    /**
     * Generate an input variable of the appropriate type and dimensionality,
     * using the given name.
     */
    input_variable_type*
    generate_input_variable(universe& u, const std::string& name = "") const;

    // Private data and methods
    //==========================================================================
  private:

    random_factor_functor<moment_gaussian>* mg_factor_func_ptr;

    random_factor_functor<canonical_gaussian>* cg_factor_func_ptr;

    random_gaussian_crf_factor_functor();

  }; // struct random_gaussian_crf_factor_functor

  //! @} group factor_random

} // namespace sill

#endif // SILL_RANDOM_GAUSSIAN_CRF_FACTOR_FUNCTOR_HPP
