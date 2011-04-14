#ifndef SILL_RANDOM_MIXTURE_CRF_FACTOR_FUNCTOR_HPP
#define SILL_RANDOM_MIXTURE_CRF_FACTOR_FUNCTOR_HPP

#include <sill/factor/mixture_crf_factor.hpp>
#include <sill/factor/random/random_crf_factor_functor_i.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random mixture_crf_factor factors.
   *
   * @tparam F  Type of CRF factor used for each mixture component.
   */
  template <typename F>
  struct random_mixture_crf_factor_functor
    : random_crf_factor_functor_i<mixture_crf_factor<F> > {

    typedef random_crf_factor_functor_i<mixture_crf_factor<F> > base;

    typedef typename base::crf_factor_type        crf_factor_type;
    typedef F                                     subfactor_type;

    typedef typename base::output_variable_type   output_variable_type;
    typedef typename base::input_variable_type    input_variable_type;

    typedef typename base::output_domain_type     output_domain_type;
    typedef typename base::input_domain_type      input_domain_type;

    typedef typename base::output_var_vector_type output_var_vector_type;
    typedef typename base::input_var_vector_type  input_var_vector_type;

    // Parameters
    //==========================================================================

    //! Number of components to create.
    //!  (default = 3)
    size_t k;

    // Public methods
    //==========================================================================

    //! Constructor.
    random_mixture_crf_factor_functor
    (size_t k, random_crf_factor_functor_i<subfactor_type>& subfactor_func)
      : k(k), subfactor_func(subfactor_func) {
    }

    using base::generate_marginal;
    using base::generate_conditional;

    /**
     * Generate a marginal factor P(Y) using the stored parameters.
     */
    crf_factor_type generate_marginal(const output_domain_type& Y) {
      std::vector<subfactor_type> comps;
      for (size_t i = 0; i < k; ++i)
        comps.push_back(subfactor_func.generate_marginal(Y));
      return mixture_crf_factor<subfactor_type>(comps);
    }

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    crf_factor_type generate_conditional(const output_domain_type& Y,
                                         const input_domain_type& X) {
      std::vector<subfactor_type> comps;
      for (size_t i = 0; i < k; ++i)
        comps.push_back(subfactor_func.generate_conditional(Y,X));
      return mixture_crf_factor<subfactor_type>(comps);
    }

    /**
     * Generate an output variable of the appropriate type and dimensionality,
     * using the given name.
     */
    output_variable_type*
    generate_output_variable(universe& u, const std::string& name = "") const {
      return subfactor_func.generate_output_variable(u, name);
    }

    /**
     * Generate an input variable of the appropriate type and dimensionality,
     * using the given name.
     */
    input_variable_type*
    generate_input_variable(universe& u, const std::string& name = "") const {
      return subfactor_func.generate_input_variable(u, name);
    }

    // Private data and methods
    //==========================================================================
  private:

    random_crf_factor_functor_i<subfactor_type>& subfactor_func;

    random_mixture_crf_factor_functor();

  }; // struct random_mixture_crf_factor_functor

  //! @} group factor_random

} // namespace sill

#endif // SILL_RANDOM_MIXTURE_CRF_FACTOR_FUNCTOR_HPP
