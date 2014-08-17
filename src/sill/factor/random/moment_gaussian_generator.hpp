#ifndef SILL_MOMENT_GAUSSIAN_GENERATOR_HPP
#define SILL_MOMENT_GAUSSIAN_GENERATOR_HPP

#include <sill/factor/moment_gaussian.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor for generating random moment_gaussian factors.
   * 
   * The functor returns a moment Gaussian, where each element of the
   * (conditional) mean is drawn from Uniform[mean_lower, mean_upper].
   * The covariance matrix is such that the variances on the diagonal
   * and the correlations between the variables are fixed.
   *
   * For conditional linear Gaussians, each entry of the coefficient
   * matrix is drawn from Uniform[coeff_lower, coeff_upper].
   *
   * \see RandomFactorGenerator
   * \ingroup factor_random
   */
  class moment_gaussian_generator {
  public:
    // RandomFactorGenerator typedefs
    typedef vector_domain   domain_type;
    typedef moment_gaussian result_type;

    struct param_type {
      double mean_lower;
      double mean_upper;
      double variance;
      double correlation;
      double coeff_lower;
      double coeff_upper;

      param_type()
        : mean_lower(-1.0),
          mean_upper(1.0),
          variance(1.0),
          correlation(0.3),
          coeff_lower(-1.0),
          coeff_upper(1.0) { }

      param_type(double mean_lower,
                 double mean_upper,
                 double variance,
                 double correlation,
                 double coeff_lower,
                 double coeff_upper)
        : mean_lower(mean_lower),
          mean_upper(mean_upper),
          variance(variance),
          correlation(correlation),
          coeff_lower(coeff_lower),
          coeff_upper(coeff_upper) {
        check();
      }

      void check() const {
        assert(variance > 0.0);
        assert(fabs(correlation) < 1.0);
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.mean_lower << " "
            << p.mean_upper << " "
            << p.variance << " "
            << p.correlation << " "
            << p.coeff_lower << " "
            << p.coeff_upper;
        return out;
      }
    }; // struct param_type

    //! Constructs a generator with the given parameters
    explicit moment_gaussian_generator(double mean_lower = -1.0,
                                       double mean_upper = +1.0,
                                       double variance = 1.0,
                                       double correlation = 0.3,
                                       double coeff_lower = -1.0,
                                       double coeff_upper = +1.0)
      : params(mean_lower, mean_upper, variance, correlation, 
               coeff_lower, coeff_upper) { }
    
    //! Constructs a generator with the given parameters
    moment_gaussian_generator(const param_type& params)
      : params(params) { }

    //! Generates a marginal distribution p(args) using the stored parameters
    template <typename RandomNumberGenerator>
    moment_gaussian operator()(const vector_domain& args,
                               RandomNumberGenerator& rng) {
      moment_gaussian result(args);
      choose_moments(rng, result);
      return result;
    }

    //! Generates a conditional distribution p(head | tail) using the stored
    //! parameters.
    template <typename RandomNumberGenerator>
    moment_gaussian operator()(const vector_domain& head,
                               const vector_domain& tail,
                               RandomNumberGenerator& rng) {
      moment_gaussian result(make_vector(head), make_vector(tail));
      choose_moments(rng, result);
      choose_coeffs(rng, result);
      return result;
    }

    //! Returns the parameter set associated with this generator
    const param_type& param() const {
      return params;
    }

    //! Sets the parameter set associated with this generator
    void param(const param_type& params) {
      params.check();
      this->params = params;
    }

  private:
    param_type params;

    template <typename RandomNumberGenerator>
    void choose_moments(RandomNumberGenerator& rng, moment_gaussian& mg) {
      boost::uniform_real<double> unif(params.mean_lower, params.mean_upper);
      foreach(double& val, mg.mean()) {
        val = unif(rng);
      }

      double covariance = params.correlation * params.variance;
      mat& cov = mg.covariance();
      cov.fill(covariance);
      cov.diag().fill(params.variance);
      if (mg.size_head() > 2 && covariance < 0.0) {
        mat tmp;
        if (!chol(tmp, cov)) {
          const char* msg =
            "moment_gaussian_generator: the correlation is too negative, "
            "the resulting covariance matrix is not PSD.";
          throw std::invalid_argument(msg);
        }
      }
    }

    template <typename RandomNumberGenerator>
    void choose_coeffs(RandomNumberGenerator& rng, moment_gaussian& mg) {
      boost::uniform_real<double> unif(params.coeff_lower, params.coeff_upper);
      foreach(double& val, mg.coefficients()) {
        val = unif(rng);
      }
    }

  }; // class moment_gaussian_generator

  //! Prints the parameters of this generator to an output stream
  //! \relates moment_gaussian_generator
  inline std::ostream&
  operator<<(std::ostream& out, const moment_gaussian_generator& gen) {
    out << gen.param();
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
