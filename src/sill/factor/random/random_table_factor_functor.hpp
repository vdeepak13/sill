#ifndef SILL_RANDOM_TABLE_FACTOR_FUNCTOR_HPP
#define SILL_RANDOM_TABLE_FACTOR_FUNCTOR_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/factor/random/random_factor_functor_i.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random table factors.
   */
  struct random_table_factor_functor
    : random_factor_functor_i<table_factor> {

    // Public types
    //==========================================================================

    typedef random_factor_functor_i<table_factor> base;

    typedef finite_variable variable_type;
    typedef finite_domain   domain_type;

    //! Parameters
    struct parameters {

      /**
       * Type of factor:
       *  - RANDOM_RANGE
       *     - Entries chosen from Uniform[lower,upper]
       *     - @see random_range_discrete_factor
       *  - ASSOCIATIVE
       *     - Diagonal factor, with 0 everywhere except along the diagonal
       *       (when Yi==Yj), in which case it has value base_val (log space).
       *     - @see make_associative_factor
       *  - RANDOM_ASSOCIATIVE
       *     - Diagonal factor, with 0 everywhere except along the diagonal
       *       (when Yi==Yj), in which case it has S.
       *       S is set via base_val + Uniform[lower,upper] (in log space).
       *     - @see make_random_associative_factor
       */
      enum factor_choice_enum { RANDOM_RANGE, ASSOCIATIVE, RANDOM_ASSOCIATIVE };

      //! Type of factor.
      //!  (default = RANDOM_RANGE)
      factor_choice_enum factor_choice;

      //! Lower bound for factor parameters (in log space).
      //!  (default = -1)
      double lower_bound;

      //! Upper bound for factor parameters (in log space).
      //!  (default = 1)
      double upper_bound;

      //! Base value used for factor parameters (in log space).
      //!  (default = 0)
      double base_val;

      //! Variable arity (> 0) used by generate_variable method.
      //!  (default = 2)
      size_t arity;

      parameters()
        : factor_choice(RANDOM_RANGE),
          lower_bound(-1), upper_bound(1), base_val(0), arity(2) { }

      //! Assert validity.
      void check() const {
        assert(factor_choice <= RANDOM_ASSOCIATIVE);
        assert(lower_bound <= upper_bound);
        assert(arity != 0);
      }

    }; // struct parameters

    // Public data
    //==========================================================================

    parameters params;

    // Public methods
    //==========================================================================

    //! Constructor.
    //! @param  random_seed  (default = time)
    explicit random_table_factor_functor(unsigned random_seed = time(NULL));

    using base::generate_marginal;
    using base::generate_conditional;

    //! Generate a marginal factor P(X) using the stored parameters.
    table_factor generate_marginal(const domain_type& X);

    //! Generate a conditional factor P(Y|X) using the stored parameters.
    //! This uses generate_marginal and then conditions on X.
    table_factor
    generate_conditional(const domain_type& Y, const domain_type& X);

    //! Generate a variable of the appropriate type and dimensionality,
    //! using the given name.
    variable_type*
    generate_variable(universe& u, const std::string& name = "") const;

    //! Set random seed.
    void seed(unsigned random_seed = time(NULL));

    // Protected data and methods
    //==========================================================================
  protected:

    boost::mt11213b rng;

  }; // struct random_table_factor_functor

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_TABLE_FACTOR_FUNCTOR_HPP
