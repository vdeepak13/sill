// Code that depends on the definitions of both moment_gaussian and
// canonical_gaussian. Not intended to be included directly.
#ifndef SILL_GAUSSIAN_COMMON_HPP
#define SILL_GAUSSIAN_COMMON_HPP

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/factor_evaluator.hpp>
#include <sill/factor/factor_mle_incremental.hpp>
#include <sill/factor/factor_sampler.hpp>

#include <boost/random/normal_distribution.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename RandomNumberGenerator>
  vector_assignment
  canonical_gaussian::sample(RandomNumberGenerator& rng) const {
    moment_gaussian mg(*this);
    return mg.sample(rng);
  }


  /**
   * Specialization of factor_evaluator for moment Gaussians.
   *
   * This class simply converts the given factor to canonical_gaussian
   * and uses the native (fast) evaluation of canonical_gaussian.
   */
  template <>
  class factor_evaluator<moment_gaussian> {
  public:
    typedef vec                 index_type;
    typedef logarithmic<double> result_type;
    typedef vector_var_vector   arg_vector_type;
    
    factor_evaluator(const moment_gaussian& mg)
      : cg(mg) { }

    logarithmic<double> operator()(const vec& arg) const {
      return cg(arg);
    }
    
    const vector_var_vector& arg_vector() const {
      return cg.arg_vector();
    }

  private:
    const canonical_gaussian cg;
  }; // class factor_evaluator<moment_gaussian>

  
  /**
   * Specialization of factor_sampler for moment Gaussians.
   *
   * This function precomputes the Cholesky decomposition of the
   * covariance matrix, so that we can draw samples from the 
   * multivariate normal efficiently.
   */
  template <>
  class factor_sampler<moment_gaussian> {
  public:
    typedef vec               index_type;
    typedef vector_var_vector var_vector_type;

    //! Creates a sampler for a marginal distribution
    factor_sampler(const moment_gaussian& factor)
      : coeff_(factor.coefficients()),
        cmean_(factor.mean()),
        trans_(chol(factor.covariance())) {
      assert(factor.marginal());
    }

    //! Creates a sampler for a conditional distribution
    factor_sampler(const moment_gaussian& factor,
                   const vector_var_vector& head)
      : coeff_(factor.coefficients()),
        cmean_(factor.mean()),
        trans_(chol(factor.covariance())) {
      assert(factor.head() == head);
    }

    //! Draws a random sample from a marginal distribution
    template <typename RandomNumberGenerator>
    void operator()(vec& sample, RandomNumberGenerator& rng) const {
      assert(coeff_.empty());
      boost::normal_distribution<> normal;
      vec vals(trans_.n_rows);
      foreach(double& val, vals) { val = normal(rng); }
      sample = cmean_ + trans_.t() * vals;
    }

    //! Draws a random sample from a conditional distribution
    template <typename RandomNumberGenerator>
    void operator()(vec& sample, const vec& tail,
                    RandomNumberGenerator& rng) const {
      boost::normal_distribution<> normal;
      vec vals(trans_.n_rows);
      foreach(double& val, vals) { val = normal(rng); }
      sample = cmean_ + coeff_ * tail + trans_.t() * vals;
    }

  private:
    mat coeff_;
    vec cmean_;
    mat trans_;

  }; // class factor_sampler<moment_gaussian>


  /**
   * Specialization of factor_mle_incremental for moment Gaussians.
   *
   * This class accumulates the first two moments of the samples.
   */
  template <>
  class factor_mle_incremental<moment_gaussian> {
  public:
    typedef double            real_type;
    typedef vector_var_vector var_vector_type;
    typedef arma::Col<double> index_type;

    struct param_type {
      double smoothing;
      param_type(double smoothing = 0.0)
        : smoothing(smoothing) { }
    };
    
    factor_mle_incremental(const vector_var_vector& args,
                           const param_type& params = param_type())
      : head_(args), weight_(0.0) {
      size_t n = vector_size(args);
      moments1_ = arma::zeros(n);
      moments2_ = arma::eye(n, n) * params.smoothing;
    }

    factor_mle_incremental(const vector_var_vector& head,
                           const vector_var_vector& tail,
                           const param_type& params = param_type())
      : head_(head), tail_(tail), weight_(0.0) {
      size_t n = vector_size(head) + vector_size(tail);
      moments1_ = arma::zeros(n);
      moments2_ = arma::eye(n, n) * params.smoothing;
    }

    void process(const arma::Col<double>& values, double weight) {
      weight_ += weight;
      moments1_ += values * weight;
      moments2_ += values * values.t() * weight;
    }

    moment_gaussian estimate() const {
      arma::Col<double> mean = moments1_ / weight_;
      arma::Mat<double> cov  = moments2_ / weight_ - mean * mean.t();
      if (tail_.empty()) {
        return moment_gaussian(head_, mean, cov);
      } else {
        return moment_gaussian(concat(tail_, head_), mean, cov)
          .conditional(make_domain(tail_));
      }
    }

    double weight() const {
      return weight_;
    }

  private:
    vector_var_vector head_;
    vector_var_vector tail_;
    arma::Col<double> moments1_; // first moments
    arma::Mat<double> moments2_; // second moments
    double weight_;

  }; // factor_mle_incremental<moment_gaussian>

  
  /**
   * Specialization of factor_mle_incremental for canonical Gaussians.
   */
  template <>
  class factor_mle_incremental<canonical_gaussian> 
    : public factor_mle_incremental<moment_gaussian> {
  public:
    typedef factor_mle_incremental<moment_gaussian> base;

    factor_mle_incremental(const vector_var_vector& args,
                           const param_type& params = param_type())
      : base(args, params) { }

    factor_mle_incremental(const vector_var_vector& head,
                           const vector_var_vector& tail,
                           const param_type& params = param_type())
      : base(head, tail, params) { }
    
    canonical_gaussian estimate() const {
      return canonical_gaussian(base::estimate());
    }
  }; // factor_mle_incremental<canoical_gaussian>
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
