#ifndef SILL_ALTERNATING_CRF_FACTOR_FUNCTOR_HPP
#define SILL_ALTERNATING_CRF_FACTOR_FUNCTOR_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/factor/random/random_crf_factor_functor_i.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Wrapper for other random_crf_factor_functor_i types which permits variety
   * via alternation between multiple random_crf_factor_functor_i instances.
   *
   * This takes two random_crf_factor_functor_i instances.
   * It generally generates factors using the first functor.
   * Every alternation_period times this functor generates a factor,
   * it generates one of those factors using the second functor.
   *
   * @tparam RFF  base random_crf_factor_functor_i type
   */
  template <typename RFF>
  struct alternating_crf_factor_functor
    : random_crf_factor_functor_i<typename RFF::crf_factor_type> {

    // Public types
    //==========================================================================

    typedef typename RFF::crf_factor_type crf_factor_type;

    typedef random_crf_factor_functor_i<crf_factor_type> base;

    typedef typename crf_factor_type::output_variable_type output_variable_type;
    typedef typename crf_factor_type::input_variable_type  input_variable_type;
    typedef typename crf_factor_type::output_domain_type   output_domain_type;
    typedef typename crf_factor_type::input_domain_type    input_domain_type;

    //! Parameters
    struct parameters {

      //! First (default) random crf_factor functor.
      //! This functor is also used for the generate_variable method.
      RFF default_rff;

      //! Second (alternate) random crf_factor functor.
      RFF alternate_rff;

      //! Alternation period (> 0)
      //! If 1, then only alternate_rff is used.
      //!  (default = 2)
      size_t alternation_period;

      parameters()
        : alternation_period(2) { }

      //! Assert validity.
      void check() const {
        assert(alternation_period != 0);
      }

    }; // struct parameters

    // Public data
    //==========================================================================

    parameters params;

    // Public methods
    //==========================================================================

    //! Constructor.
    explicit
    alternating_crf_factor_functor(unsigned random_seed = time(NULL))
      : cnt(0) {
      seed(random_seed);
    }

    using base::generate_marginal;
    using base::generate_conditional;

    //! Generate a marginal crf_factor P(X) using the stored parameters.
    crf_factor_type generate_marginal(const output_domain_type& X) {
      assert(params.alternation_period != 0);
      ++cnt;
      if (cnt % params.alternation_period == 0) {
        return params.alternate_rff.generate_marginal(X);
      } else {
        return params.default_rff.generate_marginal(X);
      }
    }

    //! Generate a conditional crf_factor P(Y|X) using the stored parameters.
    //! This uses generate_marginal and then conditions on X.
    crf_factor_type
    generate_conditional(const output_domain_type& Y,
                         const input_domain_type& X) {
      assert(params.alternation_period != 0);
      ++cnt;
      if (cnt % params.alternation_period == 0) {
        return params.alternate_rff.generate_conditional(Y,X);
      } else {
        return params.default_rff.generate_conditional(Y,X);
      }
    }

    /**
     * Generate an output variable of the appropriate type and dimensionality,
     * using the given name.
     */
    output_variable_type*
    generate_output_variable(universe& u, const std::string& name = "") const {
      return params.default_rff.generate_output_variable(u, name);
    }

    /**
     * Generate an input variable of the appropriate type and dimensionality,
     * using the given name.
     */
    input_variable_type*
    generate_input_variable(universe& u, const std::string& name = "") const {
      return params.default_rff.generate_input_variable(u, name);
    }

    //! Set random seed.
    void seed(unsigned random_seed = time(NULL)) {
      boost::mt11213b rng;
      boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
      params.default_rff.seed(unif_int(rng));
      params.alternate_rff.seed(unif_int(rng));
    }

    // Private data and methods
    //==========================================================================
  private:

    //! Counter of how many factors have been generated
    //! (for use with alternation_period).
    size_t cnt;

  }; // struct alternating_crf_factor_functor

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_ALTERNATING_CRF_FACTOR_FUNCTOR_HPP
