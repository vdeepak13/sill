#ifndef SILL_RANDOM_FACTOR_FUNCTOR_I_HPP
#define SILL_RANDOM_FACTOR_FUNCTOR_I_HPP

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Interface for functors for generating random factors.
   * Such functors can then be plugged into methods for generating random
   * models (decomposable and crf_model).
   *
   * @see create_random_crf
   */
  template <typename F>
  struct random_factor_functor_i {

    //! Factor type
    typedef F factor_type;

    //! Variable type used in the factors.
    typedef typename F::variable_type variable_type;

    //! Domain type used in the factors.
    typedef typename F::domain_type domain_type;

    virtual ~random_factor_functor_i() { }

    //! Generate a marginal factor P(X) using the stored parameters.
    virtual
    F generate_marginal(const domain_type& X) = 0;

    //! Generate a marginal factor P(X) using the stored parameters.
    F generate_marginal(variable_type* X) {
      return generate_marginal(make_domain(X));
    }

    //! Generate a conditional factor P(Y|X) using the stored parameters.
    virtual
    F generate_conditional(const domain_type& Y, const domain_type& X) = 0;

    //! Generate a conditional factor P(Y|X) using the stored parameters.
    F generate_conditional(variable_type* Y, variable_type* X) {
      return generate_conditional(make_domain(Y), make_domain(X));
    }

    //! Generate a variable of the appropriate type and dimensionality,
    //! using the given name.
    virtual
    variable_type*
    generate_variable(universe& u, const std::string& name = "") const = 0;

    //! Set random seed.
    virtual
    void seed(unsigned random_seed = time(NULL)) = 0;

  }; // struct random_factor_functor_i

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_FACTOR_FUNCTOR_I_HPP
