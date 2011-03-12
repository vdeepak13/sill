#ifndef SILL_LEARNING_GAUSSIAN_HPP
#define SILL_LEARNING_GAUSSIAN_HPP

#include <sill/factor/moment_gaussian.hpp>

namespace sill {

  // forward declaration
  template <typename LA> class dataset;

  //! \addtogroup learning_param
  //! @{

  /**
   * Computes the maximum likelihood estimate of a Gaussian distribution.
   * Each datapoint \c i in the dataset is weighted by \c w[i].
   */
  template <typename LA>
  void mle(const dataset<LA>& data, const vec& w, double regul,
           moment_gaussian& mg);

  /**
   * Computes the maximum-likelihood estimate for an unweighted dataset.
   */
  template <typename LA>
  void mle(const dataset<LA>& data, double regul, moment_gaussian& mg);

  //! @}

  //============================================================================
  // Implementation of above functions
  //============================================================================

  template <typename LA>
  void mle(const dataset<LA>& data, const vec& w, double regul,
           moment_gaussian& mg) {
    using std::log;

    size_t n = data.vector_dim();
    size_t i = 0; // datapoint index
    double sumw = sum(w);

    // compute the mean
    vec ctr(n, 0);
    foreach(const record<LA>& rec, data.records())
      ctr += w[i++] * rec.vector();
    ctr /= sumw;

    // compute the covariance
    i = 0;
    mat cov(n, n);
    cov.clear();
    foreach(const record<LA>& rec, data.records()) {
      vec x = rec.vector() - ctr;
      cov += w[i++] * outer_product(x, x);
    }
    cov /= sumw;

    // add regularization
    if (regul > 0) cov += identity(n) * regul;

    // return the resulting factor
    mg = moment_gaussian(data.vector_list(), ctr, cov, sumw);
  }

  template <typename LA>
  void mle(const dataset<LA>& data, double reg, moment_gaussian& mg) {
    mle(data, ones(data.size()), reg, mg);
  }

} // namespace sill

#endif
