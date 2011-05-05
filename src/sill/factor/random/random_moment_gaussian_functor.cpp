
#include <sill/factor/random/random_moment_gaussian_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  void
  random_moment_gaussian_functor::parameters::print(std::ostream& out) const {
    out << "b: " << b << "\n"
        << "variance: " << variance << "\n"
        << "correlation: " << correlation << "\n"
        << "c: " << c << "\n"
        << "c_shift: " << c_shift << "\n";
  }

  random_moment_gaussian_functor::
  random_moment_gaussian_functor(unsigned random_seed)
    : rng(random_seed) { }

  moment_gaussian
  random_moment_gaussian_functor::generate_marginal(const domain_type& X_) {
    params.check();
    vector_var_vector X(X_.begin(), X_.end());
    size_t Xsize = vector_size(X);
    vec mu;
    mat sigma;
    choose_mu_sigma(Xsize, mu, sigma);
    return moment_gaussian(X, mu, sigma);
  }

  moment_gaussian
  random_moment_gaussian_functor::
  generate_conditional(const domain_type& Y_, const domain_type& X_) {
    params.check();
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
  }

  void random_moment_gaussian_functor::seed(unsigned random_seed) {
    rng.seed(random_seed);
  }

  void
  random_moment_gaussian_functor::
  choose_mu_sigma(size_t Xsize, vec& mu, mat& sigma) {
    boost::uniform_real<double> unif_real(-params.b, params.b);
    mu.resize(Xsize);
    foreach(double& val, mu)
      val = unif_real(rng);
    double covariance =
      params.correlation * params.variance * params.variance;
    if (covariance == params.variance) {
      throw std::invalid_argument
        (std::string("random_moment_gaussian_functor") +
         " has variance and correlation s.t. the covariance equals" +
         " the variance, so the resulting covariance matrix is invalid.");
    }
    sigma.resize(Xsize, Xsize);
    sigma = covariance;
    for (size_t i = 0; i < Xsize; ++i)
      sigma(i,i) = params.variance;
    if (Xsize > 2) {
      if (covariance < 0 || covariance < params.variance) {
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
  random_moment_gaussian_functor::
  choose_coeff(size_t Ysize, size_t Xsize, mat& coeff) {
    boost::uniform_real<double> unif_real(-params.c, params.c);
    coeff.resize(Ysize, Xsize);
    foreach(double& val, coeff)
      val = params.c_shift + unif_real(rng);
  } // choose_coeff

  std::ostream&
  operator<<(std::ostream& out,
             const random_moment_gaussian_functor::parameters& params) {
    params.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
