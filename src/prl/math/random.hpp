#ifndef SILL_MATH_RANDOM_HPP
#define SILL_MATH_RANDOM_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/uniform_real.hpp>

#include <itpp/base/random.h>

#include <sill/math/vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  using itpp::randb;
  using itpp::randu;
  using itpp::randi;
  using itpp::randn;
  using itpp::randn_c;
  using itpp::randray;
  using itpp::randrice;
  using itpp::randexp;

  /**
   * Gamma distribution.
   * This is parameterized as Gamma(k, alpha) where the mean is k * alpha
   * and variance is k * alpha^2.
   *
   * This models a Random Distribution from Boost's Random Number Library.
   *
   * This is implemented differently for different values of k:
   *  - k >= 1: generated via Cheng and Feast's GT algorithm
   *     (Cheng, R.C.H., and G.M. Feast.  "Gamma Variate Generators with
   *      Increased Shape Parameter Range."  Communications of the ACM 23 (7),
   *      389-395 (1980).)
   *  - k < 1: generated via Best's modification to Ahren and Dieter's algorithm
   *     (Best, D.J.  "A Note on Gamma Variate Generators with Shape Parameter
   *      less than Unity."  Computing 30, 185-188 (1983).)
   *
   * \ingroup math_random
   */
  template<class RealType = double>
  class gamma_distribution {
  public:
    typedef RealType input_type;
    typedef RealType result_type;

  private:
    //! Shape parameter
    result_type k_;
    //! Scale parameter
    result_type alpha_;
    //! Uniform [0,1] distribution
    boost::uniform_real<RealType> uniform;
    //! Parameters for Cheng and Feast's GT algorithm.
    //! These double as parameters and temporary variables for Best's algorithm;
    //! the mapping to Best is:
    //!  h1->z, h2->b
    result_type a, b, c, d, h1, h2;

  public:
    //! @param k      shape parameter > 0
    //! @param alpha  scale parameter > 0
    explicit gamma_distribution(const result_type& k_, const result_type& alpha_)
      : k_(k_), alpha_(alpha_), uniform(0, 1),
        a(0), b(0), c(0), d(0), h1(0), h2(0) {
      assert(k_ > 0);
      assert(alpha_ > 0);
      if (k_ >= 1) {
        a = k_ - .5;
        b = k_ / a;
        c = 2. / a;
        d = c + 2;
        result_type s = sqrt(k_);
        h1 = (.865 + .064 / k_) / s;
        h2 = (.4343 - .105 / s) / s;
      } else { // k < 1
        h1 = .07 + .75 * sqrt(1. - k_);
        h2 = 1. + exp(-h1) * k_ / h1;
      }
    }
    //! Shape parameter
    RealType k() const { return k_; }
    //! Scale parameter
    RealType alpha() const { return alpha_; }
    //! (models Boost Random Distribution; does nothing here)
    void reset() { }
    //! Generates a random real number distributed as Gamma(k, alpha)
    template<typename Engine>
    result_type operator()(Engine& rng) {
      // Sample X ~ Gamma(k, 1) and return alpha * X
      if (k_ >= 1) {
        result_type u, u1, u2, w;
        do {
          do {
            u = uniform(rng);
            u1 = uniform(rng);
            u2 = u1 + h1 * u - h2;
          } while (u2 <= 0 || 1 <= u2);
          w = b * (u1 / u2) * (u1 / u2);
          if (c * u2 - d + w + 1. / w <= 0)
            return a * w * alpha_;
        } while (c * std::log(u2) - std::log(w) + w - 1 >= 0);
        return a * w * alpha_;
      } else { // k < 1
        result_type u, p, x, ustar, y;
        do {
          u = uniform(rng);
          p = h2 * u;
          while (p <= 1) {
            x = exp(std::log(h1) + std::log(p) / k_);
            ustar = uniform(rng);
            if (ustar <= (2. - x)/(2. + x))
              return x * alpha_;
            if (ustar <= exp(-x))
              return x * alpha_;
            u = uniform(rng);
            p = h2 * u;
          }
          x = -std::log(h1 * (h2 - p)/k_);
          y = x / h1;
          ustar = uniform(rng);
          if (ustar * (k_ + y - k_ * y) < 1)
            return x;
        } while (std::log(ustar) > (k_ - 1) * std::log(y));
        return x;
      }
    }
  }; // class gamma_distribution

  /**
   * Dirichlet(n, alpha) distribution.
   *
   * This models a Random Distribution from Boost's Random Number Library.
   *
   * \ingroup math_random
   */
  template<class RealType = double>
  class dirichlet_distribution {
  private:
    // TODO: assert RealType and vec are compatible

    //! Dimensionality of random vector
    size_t n_;
    //! Shape parameters (length 1 if all identical)
    vec alpha_;
    //! Gamma distribution
    std::vector<gamma_distribution<RealType> > gammas_;

  public:
    typedef RealType input_type;
    typedef RealType result_type;
    //! @param n      dimensionality of random vectors generated
    //! @param alpha  fixed alpha (for all n shape parameters) > 0
    explicit dirichlet_distribution(size_t n_, const result_type& alpha_)
      : n_(n_), alpha_(1, alpha_),
        gammas_(1, gamma_distribution<RealType>(alpha_, 1)) {
    }
    //! @param n      dimensionality of random vectors generated
    //! @param alpha  shape parameters alpha (n-vector) > 0
    explicit dirichlet_distribution(size_t n_, const vec& alpha_)
      : n_(n_), alpha_(alpha_) {
      assert(n_ == alpha_.size());
      for (size_t i = 0; i < n_; ++i)
        gammas_.push_back(gamma_distribution<RealType>(alpha_[i], 1));
    }
    //! dimensionality of random vector
    size_t n() const { return n_(); }
    //! Shape parameters
    const vec& alpha() const {
      if (n > alpha_.size())
        return vec(n, alpha_[0]);
      else
        return alpha_;
    }
    //! (models Boost Random Distribution; does nothing here)
    void reset() { }
    //! Generates a random vector distributed as Dirichlet(alpha)
    template<typename Engine>
    vec operator()(Engine& rng) {
      vec v(n_, 0);
      result_type total(0);
      if (n_ == alpha_.size())
        for (size_t i = 0; i < n_; ++i) {
          v[i] = gammas_[i].operator()(rng);
          total += v[i];
        }
      else
        for (size_t i = 0; i < n_; ++i) {
          v[i] = gammas_[0].operator()(rng);
          total += v[i];
        }
      for (size_t i = 0; i < n_; ++i)
        v[i] /= total;
      return v;
    }
  }; // class dirichlet distribution

}

#include <sill/macros_undef.hpp>

#endif // SILL_MATH_RANDOM_HPP
