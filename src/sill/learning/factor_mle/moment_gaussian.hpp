#ifndef SILL_FACTOR_MLE_MOMENT_GAUSSIAN_HPP
#define SILL_FACTOR_MLE_MOMENT_GAUSSIAN_HPP

#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/vector_record.hpp>
#include <sill/learning/factor_mle/factor_mle.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // moment gaussian maximum likelihood estimator
  // eventually: add the template argument for the storage type of the factor
  template <>
  class factor_mle<moment_gaussian> {
  public:
    typedef vector_dataset<> dataset_type;
    typedef vector_domain    domain_type;

    struct param_type {
      double smoothing;
      param_type(double smoothing = 0.0) // intentionally implicit
        : smoothing(smoothing) { }
    };

    factor_mle(const vector_dataset<>* dataset,
        const param_type& params = param_type()) 
      : dataset(dataset), params(params) {
      assert(dataset->size() > 0);
    }

    //! Returns the marginal distribution over a subset of variables
    moment_gaussian operator()(const vector_domain& vars) const {
      return operator()(make_vector(vars));
    }

    //! Returns the marginal distribution over a sequence of variables
    moment_gaussian operator()(const vector_var_vector& vars) const {
      size_t n = vector_size(vars);

      // compute the mean
      vec mean = arma::zeros(n);
      double sumw = 0.0;
      foreach(const vector_record<double>& r, dataset->records(vars)) {
        mean += r.weight * r.values;
        sumw += r.weight;
      }
      mean /= sumw;

      // compute the covariance
      mat cov = arma::zeros(n, n);
      foreach(const vector_record<double>& r, dataset->records(vars)) {
        vec x = r.values - mean;
        cov += r.weight * (x * x.t());
      }
      cov /= sumw;

      // add regularization
      if (params.smoothing > 0.0) {
        cov += arma::eye(n,n) * params.smoothing;
      }

      return moment_gaussian(vars, mean, cov);
    }

    //! Returns the marginal distribution over a subset of variables
    //! reweighing the dataset
    moment_gaussian operator()(const vector_domain& vars,
                               const vec& weights) const {
      return operator()(make_vector(vars), weights);
    }

    //! Returns the marginal distribution over a sequence of variables
    //! reweighing the dataset
    moment_gaussian operator()(const vector_var_vector& vars,
                               const vec& weights) const {
      assert(weights.size() == dataset->size());
      size_t n = vector_size(vars);

      // compute the mean
      vec mean = arma::zeros(n);
      double sumw = 0.0;
      size_t i = 0;
      foreach(const vector_record<double>& r, dataset->records(vars)) {
        mean += (r.weight * weights[i]) * r.values;
        sumw += (r.weight * weights[i]);
        ++i;
      }
      mean /= sumw;

      // compute the covariance
      mat cov = arma::zeros(n, n);
      i = 0;
      foreach(const vector_record<double>& r, dataset->records(vars)) {
        vec x = r.values - mean;
        cov += (r.weight * weights[i]) * (x * x.t());
        ++i;
      }
      cov /= sumw;

      // add regularization
      if (params.smoothing > 0.0) {
        cov += arma::eye(n,n) * params.smoothing;
      }

      return moment_gaussian(vars, mean, cov, sumw);
    }

  private:
    const vector_dataset<>* dataset;
    param_type params;

  }; // factor_mle<moment_gaussian>

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
