#ifndef SILL_RANDOM_ALTERNATING_FACTOR_FUNCTOR_HPP
#define SILL_RANDOM_ALTERNATING_FACTOR_FUNCTOR_HPP

#include <sill/factor/random/random_factor_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Wrapper for other random_factor_functor types which
   * permits variety via alternation between multiple random_factor_functor
   * instances.
   *
   * This takes two random_factor_functors.
   * It generally generates factors using the first functor.
   * Every alternation_period times this functor generates a factor,
   * it generates one of those factors using the second functor.
   *
   * @tparam RFF  base random_factor_functor type
   */
  template <typename RFF>
  struct random_alternating_factor_functor
    : random_factor_functor<typename RFF::factor_type> {

    typedef typename RFF::factor_type factor_type;

    typedef random_factor_functor<factor_type> base;

    typedef typename factor_type::variable_type variable_type;
    typedef typename factor_type::domain_type   domain_type;

    // Parameters
    //==========================================================================

    //! First (default) random factor functor.
    //! This functor is also used for the generate_variable method.
    RFF default_rff;

    //! Second (alternate) random factor functor.
    RFF alternate_rff;

    //! Alternation period (> 0)
    //! If 1, then only alternate_rff is used.
    //!  (default = 2)
    size_t alternation_period;

    // Public methods
    //==========================================================================

    //! Constructor.
    random_alternating_factor_functor(RFF default_rff,
                                      RFF alternate_rff)
      : default_rff(default_rff), alternate_rff(alternate_rff),
        alternation_period(2), cnt(0) { }

    using base::generate_marginal;
    using base::generate_conditional;

    //! Generate a marginal factor P(X) using the stored parameters.
    factor_type generate_marginal(const domain_type& X) {
      assert(alternation_period != 0);
      ++cnt;
      if (cnt % alternation_period == 0) {
        return alternate_rff.generate_marginal(X);
      } else {
        return default_rff.generate_marginal(X);
      }
    }

    //! Generate a conditional factor P(Y|X) using the stored parameters.
    //! This uses generate_marginal and then conditions on X.
    factor_type
    generate_conditional(const domain_type& Y, const domain_type& X) {
      assert(alternation_period != 0);
      ++cnt;
      if (cnt % alternation_period == 0) {
        return alternate_rff.generate_conditional(Y,X);
      } else {
        return default_rff.generate_conditional(Y,X);
      }
    }

    //! Generate a variable of the appropriate type and dimensionality,
    //! using the given name.
    variable_type*
    generate_variable(universe& u, const std::string& name = "") const {
      return default_rff.generate_variable(u, name);
    }

    // Private data and methods
    //==========================================================================
  private:

    //! Counter of how many factors have been generated
    //! (for use with alternation_period).
    size_t cnt;

    random_alternating_factor_functor();

    random_alternating_factor_functor
    (const random_alternating_factor_functor& other);

  }; // struct random_alternating_factor_functor

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_ALTERNATING_FACTOR_FUNCTOR_HPP
