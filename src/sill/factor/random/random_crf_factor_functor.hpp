#ifndef SILL_RANDOM_CRF_FACTOR_FUNCTOR_HPP
#define SILL_RANDOM_CRF_FACTOR_FUNCTOR_HPP

#include <sill/base/universe.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Interface for functors for generating random crf_factor factors.
   * Such functors can then be plugged into methods for generating random
   * models (decomposable and crf_model).
   *
   * @see create_random_crf
   */
  template <typename CRFfactor>
  struct random_crf_factor_functor {

    //! Factor type
    typedef CRFfactor crf_factor_type;

    //! Type of Y in the conditional P(Y|X) this functor can generate.
    typedef typename CRFfactor::output_variable_type output_variable_type;

    //! Type of X in the conditional P(Y|X) this functor can generate.
    typedef typename CRFfactor::input_variable_type input_variable_type;

    //! Y domain type used in the factors P(Y|X).
    typedef typename CRFfactor::output_domain_type output_domain_type;

    //! X domain type used in the factors P(Y|X).
    typedef typename CRFfactor::input_domain_type input_domain_type;

    //! Y vector type used in the factors P(Y|X).
    typedef typename CRFfactor::output_var_vector_type output_var_vector_type;

    //! X vector type used in the factors P(Y|X).
    typedef typename CRFfactor::input_var_vector_type input_var_vector_type;

    // Public methods
    //==========================================================================

    /**
     * Generate a marginal factor P(Y) using the stored parameters.
     */
    virtual crf_factor_type generate_marginal(const output_domain_type& Y) = 0;

    /**
     * Generate a marginal factor P(Y) using the stored parameters.
     */
    crf_factor_type generate_marginal(input_variable_type* Y) {
      return generate_marginal(make_domain(Y));
    }

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    virtual
    crf_factor_type generate_conditional(const output_domain_type& Y,
                                         const input_domain_type& X) = 0;

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    crf_factor_type generate_conditional(output_variable_type* Y,
                                         input_variable_type* X) {
      return generate_conditional(make_domain(Y), make_domain(X));
    }

    /**
     * Generate an output variable of the appropriate type and dimensionality,
     * using the given name.
     */
    virtual
    output_variable_type*
    generate_output_variable(universe& u, const std::string& name = "") const=0;

    /**
     * Generate an input variable of the appropriate type and dimensionality,
     * using the given name.
     */
    virtual
    input_variable_type*
    generate_input_variable(universe& u, const std::string& name = "") const =0;

  }; // struct random_crf_factor_functor

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_CRF_FACTOR_FUNCTOR_HPP
