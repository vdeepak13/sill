#ifndef SILL_RANDOM_TABLE_CRF_FACTOR_FUNCTOR_HPP
#define SILL_RANDOM_TABLE_CRF_FACTOR_FUNCTOR_HPP

#include <sill/factor/table_crf_factor.hpp>
#include <sill/factor/random/random_crf_factor_functor_i.hpp>
#include <sill/factor/random/random_table_factor_functor.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random table_crf_factor factors.
   */
  struct random_table_crf_factor_functor
    : random_crf_factor_functor_i<table_crf_factor> {

    // Public types
    //==========================================================================

    typedef random_crf_factor_functor_i<table_crf_factor> base;

    //! Parameters
    struct parameters {

      random_table_factor_functor table_factor_func;

      //! Assert validity.
      void check() const {
        table_factor_func.params.check();
      }

    }; // struct parameters

    // Public data
    //==========================================================================

    parameters params;

    // Public methods
    //==========================================================================

    //! Constructor.
    explicit
    random_table_crf_factor_functor(unsigned random_seed = time(NULL));

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

    //! Set random seed.
    void seed(unsigned random_seed = time(NULL));

  }; // struct random_table_crf_factor_functor

  //! @} group factor_random

} // namespace sill

#endif // SILL_RANDOM_TABLE_CRF_FACTOR_FUNCTOR_HPP
