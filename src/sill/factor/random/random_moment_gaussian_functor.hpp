#ifndef SILL_RANDOM_MOMENT_GAUSSIAN_FUNCTOR_HPP
#define SILL_RANDOM_MOMENT_GAUSSIAN_FUNCTOR_HPP

#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/random/random_factor_functor_i.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random moment_gaussian factors.
   */
  struct random_moment_gaussian_functor
    : random_factor_functor_i<moment_gaussian> {

    typedef random_factor_functor_i<moment_gaussian> base;

    typedef vector_variable variable_type;
    typedef vector_domain   domain_type;

    //! Parameters
    struct parameters {

      // Parameters for head variables, i.e., for Y in P(Y) or P(Y|X).
      //========================================================================

      //! Each element of the mean is chosen from Uniform[-b, b].  (b >= 0)
      //!  (default = 1)
      double b;

      //! Set variances of each variable to this value.  (variance > 0)
      //!  (default = 1)
      double variance;

      //! Set covariance of each pair of variables according to this correlation
      //! coefficient.  (fabs(correlation) <= 1)
      //!  (default = .3)
      double correlation;

      // Parameters for tail variables, i.e., for X in P(Y|X).
      //========================================================================

      //! Each element of the coefficient matrix C is chosen
      //! from c_shift + Uniform[-c, c],
      //! where C shifts the mean when conditioning on X=x.  (c >= 0)
      //!  (default = 1)
      double c;

      //! Each element of the coefficient matrix C is chosen
      //! from c_shift + Uniform[-c, c],
      //! where C shifts the mean when conditioning on X=x.
      //!  (default = 0)
      double c_shift;

      parameters()
        : b(1), variance(1), correlation(.3), c(1), c_shift(0) { }

      //! Assert validity.
      void check() const {
        assert(b >= 0);
        assert(variance > 0);
        assert(fabs(correlation) <= 1);
        assert(c >= 0);
      }

      //! Print the options in this struct.
      void print(std::ostream& out) const;

    }; // struct parameters

    // Public data
    //==========================================================================

    parameters params;

    // Public methods
    //==========================================================================

    //! Constructor.
    explicit random_moment_gaussian_functor(unsigned random_seed = time(NULL));

    using base::generate_marginal;
    using base::generate_conditional;

    /**
     * Generate a marginal factor P(X) using the stored parameters.
     */
    moment_gaussian generate_marginal(const domain_type& X);

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    moment_gaussian
    generate_conditional(const domain_type& Y, const domain_type& X);

    //! Generate a variable of the appropriate type and dimensionality,
    //! using the given name.
    variable_type*
    generate_variable(universe& u, const std::string& name = "") const {
      return u.new_vector_variable(name, 1);
    }

    //! Set random seed.
    void seed(unsigned random_seed = time(NULL));

    // Private data and methods
    //==========================================================================
  private:

    boost::mt11213b rng;

    void choose_mu_sigma(size_t Xsize, vec& mu, mat& sigma);

    void choose_coeff(size_t Ysize, size_t Xsize, mat& coeff);

  }; // struct random_moment_gaussian_functor

  std::ostream&
  operator<<(std::ostream& out,
             const random_moment_gaussian_functor::parameters& params);

  //! @} group factor_random

} // namespace sill

#endif // SILL_RANDOM_MOMENT_GAUSSIAN_FUNCTOR_HPP
