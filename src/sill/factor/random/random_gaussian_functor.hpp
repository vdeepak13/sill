#ifndef SILL_RANDOM_GAUSSIAN_FUNCTOR_HPP
#define SILL_RANDOM_GAUSSIAN_FUNCTOR_HPP

#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/random/random_factor_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  /**
   * Functor for generating random Gaussian factors.
   * This functor's parameters are in terms of a moment representation.
   *
   * @tparam F   Gaussian factor type.
   */
  template <typename F>
  struct random_gaussian_functor
    : random_factor_functor<F> {

    typedef random_factor_functor<F> base;

    typedef typename base::variable_type variable_type;
    typedef typename base::domain_type   domain_type;

    // Parameters for head variables, i.e., for Y in P(Y) or P(Y|X).
    //==========================================================================

    //! Each element of the mean is chosen from Uniform[-b, b].
    //!  (default = 1)
    double b;

    //! Set variances of each variable to this value.
    //!  (default = 1)
    double variance;

    //! Set covariance of each pair of variables according to this correlation
    //! coefficient.
    //!  (default = .3)
    double correlation;

    // Parameters for tail variables, i.e., for X in P(Y|X).
    //==========================================================================

    //! Each element of the coefficient matrix C is chosen
    //! from c_shift + Uniform[-c, c],
    //! where C shifts the mean when conditioning on X=x.
    //!  (default = 1)
    double c;

    //! Each element of the coefficient matrix C is chosen
    //! from c_shift + Uniform[-c, c],
    //! where C shifts the mean when conditioning on X=x.
    //!  (default = 0)
    double c_shift;

    // Public methods
    //==========================================================================

    //! Constructor.
    explicit random_gaussian_functor(unsigned random_seed)
      : b(1), variance(1), correlation(.3), c(1), c_shift(0),
        rng(random_seed) { }

    using base::generate_marginal;
    using base::generate_conditional;

    /**
     * Generate a marginal factor P(X) using the stored parameters.
     */
    F generate_marginal(const domain_type& X) {
      return F(generate_marginal_mg(X));
    }

    /**
     * Generate a conditional factor P(Y|X) using the stored parameters.
     */
    F generate_conditional(const domain_type& Y, const domain_type& X) {
      return F(generate_conditional_mg(Y,X));
    }

    //! Generate a variable of the appropriate type and dimensionality,
    //! using the given name.
    variable_type*
    generate_variable(universe& u, const std::string& name = "") const {
      return u.new_vector_variable(name, 1);
    }

    // Private data and methods
    //==========================================================================
  private:

    boost::mt11213b rng;

    random_gaussian_functor();

    random_gaussian_functor(const random_gaussian_functor& other);

    moment_gaussian generate_marginal_mg(const domain_type& X_) {
      vector_var_vector X(X_.begin(), X_.end());
      size_t Xsize = vector_size(X);
      vec mu;
      mat sigma;
      choose_mu_sigma(Xsize, mu, sigma);
      return moment_gaussian(X, mu, sigma);
    } // generate_marginal_mg

    moment_gaussian
    generate_conditional_mg(const domain_type& Y_, const domain_type& X_) {
      vector_var_vector Y(Y_.begin(), Y_.end());
      size_t Ysize = vector_size(Y);
      vec mu;
      mat sigma;
      choose_mu_sigma(Ysize, mu, sigma);
      vector_var_vector X(X_.begin(), X_.end());
      size_t Xsize = vector_size(X);
      mat coeff;
      choose_coeff(Ysize, Xsize, coeff);
      return moment_gaussian(Y, mu, sigma, X, coeff);
    } // generate_conditional_mg

    void
    choose_mu_sigma(size_t Xsize, vec& mu, mat& sigma) {
      if ((b < 0) || (variance <= 0) || (fabs(correlation) > 1)) {
        throw std::invalid_argument
          (std::string("random_moment_gaussian_functor") +
           " has invalid parameters.");
      }
      boost::uniform_real<double> unif_real(-b, b);
      mu.resize(Xsize);
      foreach(double& val, mu)
        val = unif_real(rng);
      double covariance = correlation * variance * variance;
      if (covariance == variance) {
        throw std::invalid_argument
          (std::string("random_moment_gaussian_functor") +
           " has variance and correlation s.t. the covariance equals" +
           " the variance, so the resulting covariance matrix is invalid.");
      }
      sigma.resize(Xsize, Xsize);
      sigma = covariance;
      for (size_t i = 0; i < Xsize; ++i)
        sigma(i,i) = variance;
      if (Xsize > 2) {
        if (covariance < 0 || covariance < variance) {
          mat tmpmat;
          bool result = chol(sigma, tmpmat);
          if (!result) {
            throw std::invalid_argument
              (std::string("random_moment_gaussian_functor has variance") +
               " and correlation s.t. the covariance matrix is invalid.");
          }
        }
      }
    } // choose_mu_sigma

    void
    choose_coeff(size_t Ysize, size_t Xsize, mat& coeff) {
      if (c < 0) {
        throw std::invalid_argument
          (std::string("random_moment_gaussian_functor") +
           " has invalid parameters.");
      }
      boost::uniform_real<double> unif_real(-c, c);
      coeff.resize(Ysize, Xsize);
      foreach(double& val, coeff)
        val = c_shift + unif_real(rng);
    } // choose_coeff

  }; // struct random_gaussian_functor

  //! Specialization for moment_gaussian
  template <>
  moment_gaussian
  random_gaussian_functor<moment_gaussian>::
  generate_marginal(const domain_type& X);

  template <>
  moment_gaussian
  random_gaussian_functor<moment_gaussian>::
  generate_conditional(const domain_type& Y, const domain_type& X);

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RANDOM_GAUSSIAN_FUNCTOR_HPP
