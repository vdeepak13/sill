#ifndef SILL_LEARNING_GAUSSIAN_HPP
#define SILL_LEARNING_GAUSSIAN_HPP

#include <sill/factor/moment_gaussian.hpp>

namespace sill {

  // forward declaration
  class dataset;

  //! \addtogroup learning_param
  //! @{

  /**
   * Computes the maximum likelihood estimate of a Gaussian distribution.
   * Each datapoint \c i in the dataset is weighted by \c w[i].
   */
  void mle(const dataset& data, const vec& w, double regul,moment_gaussian& mg);

  /**
   * Computes the maximum-likelihood estimate for an unweighted dataset.
   */
  void mle(const dataset& data, double regul, moment_gaussian& mg);

  //! @}

} // namespace sill

#endif
