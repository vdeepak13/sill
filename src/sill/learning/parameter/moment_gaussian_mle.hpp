#ifndef SILL_MOMENT_GAUSSIAN_MLE_HPP
#define SILL_MOMENT_GAUSSIAN_MLE_HPP

#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/dataset2/vector_dataset.hpp>
#include <sill/learning/dataset2/vector_record.hpp>
#include <sill/learning/parameter/factor_estimator.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // moment gaussian maximum likelihood estimator
  // eventually: add the template argument for the storage type of the factor
  template <typename Dataset=vector_dataset<> >
  class moment_gaussian_mle : public factor_estimator<moment_gaussian> {
  public:
    moment_gaussian_mle(const Dataset* dataset, double smoothing = 0.0)
      : dataset(dataset), smoothing(smoothing) {
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
      foreach(const vector_record2<double>& r, dataset->records(vars)) {
        mean += r.weight * r.values;
        sumw += r.weight;
      }
      mean /= sumw;

      // compute the covariance
      mat cov = arma::zeros(n, n);
      foreach(const vector_record2<double>& r, dataset->records(vars)) {
        vec x = r.values - mean;
        cov += r.weight * (x * x.t());
      }
      cov /= sumw;

      // add regularization
      if (smoothing > 0.0) {
        cov += arma::eye(n,n) * smoothing;
      }

      return moment_gaussian(vars, mean, cov);
    }

    //! Returns the marginal distribution over a subset of variables,
    //! reweighting the dataset
    moment_gaussian operator()(const vector_domain& vars,
                               const vec& weights) const {
      return operator()(make_vector(vars), weights);
    }

    //! Returns the marginal distribution over a sequence of variables,
    //! reweighting the dataset
    moment_gaussian operator()(const vector_var_vector& vars,
                               const vec& weights) const {
      assert(weights.size() == dataset->size());
      size_t n = vector_size(vars);

      // compute the mean
      vec mean = arma::zeros(n);
      double sumw = 0.0;
      size_t i = 0;
      foreach(const vector_record2<double>& r, dataset->records(vars)) {
        mean += (r.weight * weights[i]) * r.values;
        sumw += (r.weight * weights[i]);
        ++i;
      }
      mean /= sumw;

      // compute the covariance
      mat cov = arma::zeros(n, n);
      i = 0;
      foreach(const vector_record2<double>& r, dataset->records(vars)) {
        vec x = r.values - mean;
        cov += (r.weight * weights[i]) * (x * x.t());
        ++i;
      }
      cov /= sumw;

      // add regularization
      if (smoothing > 0.0) {
        cov += arma::eye(n,n) * smoothing;
      }

      return moment_gaussian(vars, mean, cov, sumw);
    }

  private:
    const Dataset* dataset;
    double smoothing;

  }; // class moment_gaussian_mle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
