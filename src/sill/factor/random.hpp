#ifndef SILL_RANDOM_FACTOR_HPP
#define SILL_RANDOM_FACTOR_HPP

#include <boost/random/uniform_real.hpp>

#include <sill/base/finite_assignment.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/math/random.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup factor_random
  //! @{

  // Marginal (non-conditional) factors for finite variables
  //============================================================================

  /**
   * Creates a factor \f$\phi\f$ over a single binary variable \f$x\f$
   * such that \f$\phi(x)=\exp(\alpha)\f$ for \f$x=1\f$ 
   * and \f$\phi(x)=\exp(-\alpha)\f$ for \f$x=0\f$.
   */
  template <typename F>
  F make_ising_factor(finite_variable* u, double alpha) {
    using std::exp;
    assert(u->size() == 2);
    F f(make_domain(u), 0);
    finite_assignment a;
    a[u] = 0; f.set_logv(a, -alpha);
    a[u] = 1; f.set_logv(a, +alpha);
    return f;
  }

  //! Creates an Ising factor f over two variables, i.e.,
  //! f(u,v)=1 if u=v and exp(-lambda) otherwise.
  template <typename F>
  F make_ising_factor(finite_variable* u, finite_variable* v, double lambda) {
    using namespace std; // for exp lookup
    assert(u->size() == v->size());
    F f(make_domain(u, v), exp(-lambda));
    for(size_t i = 0; i < u->size(); i++) {
      f.set_logv(i,i, lambda);
    }
    return f;
  }

  //! Creates a factor of an Ising model with the lambda drawn
  template <typename F, typename Engine>
  F random_ising_factor(finite_variable* u, finite_variable* v, Engine& engine){
    return make_ising_factor<F>(u, v, boost::uniform_real<double>()(engine));
  }

  //! Contructs a discrete factor with entries filled uniformly between 0 and 1
  //! with the given default value and fill factor.
  template <typename F, typename Engine>
  F random_discrete_factor(typename F::domain_type args, 
                           Engine& engine,
                           double default_value,
                           double fill_factor) {
    concept_assert((Factor<F>)); // should be DiscreteFactor
    F f(args, default_value);
    boost::uniform_real<double> rand; // random number between 0 and 1
    size_t n = static_cast<size_t>(f.size() * fill_factor);
    for (size_t i = 0; i < n; i++)
      f.set_v(random_assignment(args, engine), rand(engine));
    return f;
  }

  //! Constructs a discrete factor with entries filled uniformly between 0 and 1
  template <typename F, typename Engine>
  F random_discrete_factor(typename F::domain_type args, Engine& engine) {
    concept_assert((Factor<F>)); // should be DiscreteFactor
    F f(args, 0);
    boost::uniform_real<double> rand;
    foreach(finite_assignment a, f.assignments()) f.set_v(a, rand(engine));
    return f;
  }

  //! Constructs a discrete factor with entries filled uniformly between
  //! lower and upper bounds.
  template <typename F, typename Engine>
  F random_range_discrete_factor(typename F::domain_type args, 
                                 Engine& engine,
                                 double lower,
                                 double upper) {
    concept_assert((Factor<F>)); // should be DiscreteFactor
    F f(args, 0);
    boost::uniform_real<double> rand(lower, upper);
    foreach(double& value, f.values()) value = rand(engine);
    return f;
  }

  /**
   * Creates a factor over the given discrete variables which is diagonal;
   * i.e., it has 0 everywhere except for when Yi == Yj, in which case
   * it has 's'.
   */
  table_factor
  make_associative_factor(finite_variable* Yi, finite_variable* Yj, double s);

  /**
   * Creates a factor over the given variables which is diagonal;
   * i.e., it has 0 everywhere except for when Yi == Yj, in which case
   * it has 's'.
   * 's' is set to be exp(base + Uniform[-range, range]) for each element of
   * the diagonal.
   */
  template <typename Engine>
  table_factor
  make_random_associative_factor(finite_variable* Yi, finite_variable* Yj,
                                 double base, double range,
                                 Engine& rng) {
    assert(Yi && Yj);
    assert(Yi->size() == Yj->size());
    assert(range >= 0);
    table_factor f(make_domain<finite_variable>(Yi,Yj), 1.);
    finite_assignment fa;
    boost::uniform_real<double> unif_real(-range, range);
    for (size_t k(0); k < Yi->size(); ++k) {
      fa[Yi] = k;
      fa[Yj] = k;
      f(fa) = std::exp(base + unif_real(rng));
    }
    return f;
  }

  // Marginal (non-conditional) factors for vector variables
  //============================================================================

  /**
   * Constructs a marginal Gaussian P(A,B) over 2 variables A,B:
   *  - Chooses a random mean in [-b_max, b_max] for each variable.
   *  - Sets the variances of A,B to variance; sets the covariance of A,B
   *    according to the given correlation coefficient.
   * @param b_max         Value >= 0.
   * @param variance      Value > 0.
   * @param correlation   Value in [-1, 1].
   */
  template <typename Engine>
  moment_gaussian
  make_binary_marginal_gaussian(vector_variable* A, vector_variable* B,
                                double b_max, double variance,
                                double correlation,
                                Engine& rng) {
    if ((b_max < 0) || (variance <= 0) || (fabs(correlation) > 1))
      throw std::invalid_argument
        ("make_binary_marginal_gaussian() given invalid argument.");
    boost::uniform_real<double> unif_real(-b_max, b_max);
    vector_var_vector X;
    X.push_back(A);
    X.push_back(B);
    vec mu(2, 0);
    foreach(double& val, mu)
      val = unif_real(rng);
    mat sigma(2, 2, variance);
    double covariance = correlation * variance * variance;
    if (covariance == variance) {
      throw std::invalid_argument
        (std::string("make_binary_marginal_gaussian") +
         " given variance and correlation s.t. the covariance equals" +
         " the variance, so the resulting covariance matrix is invalid.");
    }
    sigma(0,1) = covariance;
    sigma(1,0) = covariance;
    return moment_gaussian(X, mu, sigma);
  }

  /**
   * Constructs a marginal Gaussian P(X) as follows:
   *  - Chooses a random mean in [-b_max, b_max] for each variable.
   *  - Chooses a covariance matrix:
   *      spread * Identity + cov_strength * (matrix of ones).
   * @param b_max         Value >= 0.
   * @param spread        Value > 0.
   * @param cov_strength  Value >= 0.
   */
  template <typename Engine>
  moment_gaussian
  make_marginal_gaussian_factor(const vector_var_vector& X, double b_max,
                                double spread, double cov_strength,
                                Engine& rng) {
    if ((b_max < 0) || (spread <= 0) || (cov_strength < 0))
      throw std::invalid_argument
        ("make_marginal_gaussian_factor() given invalid argument.");
    boost::uniform_real<double> unif_real(-b_max, b_max);
    size_t Xsize(vector_size(X));
    vec mu(Xsize, 0);
    foreach(double& val, mu)
      val = unif_real(rng);
    mat sigma(Xsize, Xsize, cov_strength);
    sigma += spread * identity(Xsize);
    return moment_gaussian(X, mu, sigma);
  }

  // Conditional factors for finite variables
  //============================================================================

  //! Constructs a discrete conditional CPT P( Y | X\Y ) with the distribution
  //! over Y filled using a Dirichlet(alpha).
  template <typename F, typename Engine>
  F random_discrete_conditional_factor(typename F::domain_type Y,
                                       typename F::domain_type X,
                                       double alpha, Engine& engine) {
    concept_assert((Factor<F>)); // should be DiscreteFactor
    F f(set_union(Y, X), 0);
    typename F::domain_type given(set_difference(X, Y));
    size_t card_Y(num_assignments(Y));
    dirichlet_distribution<double> dirichlet(card_Y, alpha);
    foreach(const finite_assignment& aX, assignments(given)) {
      vec vals(dirichlet(engine));
      size_t i(0);
      foreach(const finite_assignment& aY, assignments(Y)) {
        f.set_v(map_union(aX, aY), vals[i]);
        ++i;
      }
    }
    return f;
  }

  /**
   * Helper method for create_random_crf().
   * This creates an edge factor whose type is determined by 'factor_choice':
   *  - "random"
   *  - "associative"
   *  - "random_assoc"
   */
  template <typename Engine>
  table_factor
  create_random_crf_table_factor(const std::string& factor_choice,
                                 finite_variable* y1, finite_variable* y2,
                                 Engine& rng, double strength,
                                 double strength_base) {
    if (factor_choice == "random") {
      table_factor f(random_range_discrete_factor<table_factor>
                     (make_domain<finite_variable>(y1,y2), rng,
                      -strength, strength));
      f.update(exponent<double>());
      return f;
    } else if (factor_choice == "associative")
      return table_factor(make_associative_factor(y1,y2, std::exp(strength)));
    else if (factor_choice == "random_assoc")
      return table_factor(make_random_associative_factor
                          (y1, y2, strength_base, strength, rng));
    else
      throw std::invalid_argument("bad factor_choice: " + factor_choice);
  }

  // Conditional factors for vector variables
  //============================================================================

  /**
   * Constructs a conditional Gaussian P(Y|X) as follows:
   *  - Chooses a mean from Uniform[-b_max,b_max] for each variable.
   *  - Chooses a random coefficient matrix C (for computing the mean
   *    conditioned on X=x), with each element chosen from
   *    Uniform[-c_max,c_max].
   *  - Chooses a covariance matrix:
   *      spread * Identity + cov_strength * (matrix of ones).
   * @param b_max         Value >= 0.
   * @param c_max         Value > 0.
   * @param spread        Value > 0.
   * @param cov_strength  Value >= 0.
   */
  template <typename RandomNumberGenerator>
  moment_gaussian
  make_conditional_gaussian_factor(const vector_var_vector& Y,
                                   const vector_var_vector& X,
                                   double b_max, double c_max,
                                   double spread, double cov_strength,
                                   RandomNumberGenerator& rng) {
    if ((b_max < 0) || (c_max <= 0) || (spread <= 0) || (cov_strength < 0))
      throw std::invalid_argument
        ("make_conditional_gaussian_factor() given invalid argument.");
    boost::uniform_real<double> unif_real(-b_max, b_max);
    size_t Ysize(vector_size(Y));
    size_t Xsize(vector_size(X));
    vec mu(Ysize, 0);
    foreach(double& val, mu)
      val = unif_real(rng);
    unif_real = boost::uniform_real<double>(-c_max, c_max);
    mat coeff(Ysize, Xsize, 0.);
    for (size_t j(0); j < coeff.size(); ++j)
      coeff(j) = unif_real(rng);
    mat sigma(Ysize, Ysize, cov_strength);
    sigma += spread * identity(Ysize);
    return moment_gaussian(Y, mu, sigma, X, coeff);
  }

  /**
   * Constructs a conditional Gaussian P(Y|X) where Y,X are individual variables
   * as follows:
   *  - Chooses a mean from Uniform[-b_max,b_max] for Y.
   *  - Chooses a random coefficient C (for computing the mean
   *    conditioned on X=x), chosen from Uniform[-c_max,c_max].
   * @param b_max         Value >= 0.
   * @param c_max         Value > 0.
   */
  template <typename RandomNumberGenerator>
  moment_gaussian
  make_binary_conditional_gaussian
  (vector_variable* Y, vector_variable* X, double b_max, double c_max,
   RandomNumberGenerator& rng) {
    if ((b_max < 0) || (c_max <= 0))
      throw std::invalid_argument
        ("make_binary_conditional_gaussian_factor() given invalid argument.");
    boost::uniform_real<double> unif_real(-b_max, b_max);
    size_t Ysize(Y->size());
    size_t Xsize(X->size());
    assert(Ysize == 1);
    assert(Xsize == 1);
    vec mu(Ysize, 0);
    foreach(double& val, mu)
      val = unif_real(rng);
    unif_real = boost::uniform_real<double>(-c_max, c_max);
    mat coeff(Ysize, Xsize, 0.);
    for (size_t j(0); j < coeff.size(); ++j)
      coeff(j) = unif_real(rng);
    mat sigma(Ysize, Ysize, 1.);
    return moment_gaussian(vector_var_vector(1,Y), mu, sigma,
                           vector_var_vector(1,X), coeff);
  }

  //! @} group factor_random

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
