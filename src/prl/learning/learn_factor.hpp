
#ifndef SILL_LEARN_FACTOR_HPP
#define SILL_LEARN_FACTOR_HPP

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/dataset.hpp>

#include <sill/macros_def.hpp>

/**
 * \file learn_factor.hpp Methods for learning regular (non-CRF) factors.
 *
 * @todo Generalize these methods as I did for learning CRF factors so that
 *       they can take a number of lambdas specified via the factor type.
 */

namespace sill {

  /**
   * Learns a marginal factor P(X) from data.
   * @param X          Variables in marginal.
   * @param ds         Training data.
   * @param smoothing  Regularization (>= 0).
   */
  template <typename F>
  F
  learn_marginal(const typename F::domain_type& X, const dataset& ds,
                 double smoothing);

  /**
   * Learns a marginal factor P(X) from data, using no regularization.
   * @param X          Variables in marginal.
   * @param ds         Training data.
   */
  template <typename F>
  F
  learn_marginal(const typename F::domain_type& X, const dataset& ds) {
    return learn_marginal<F>(X, ds, 0);
  }

  /**
   * Learns a conditional factor P(A | B, C=c) from data.
   * @param ds         Training data.
   * @param smoothing  Regularization (>= 0).
   */
  template <typename F>
  F
  learn_conditional(const typename F::domain_type& A,
                    const typename F::domain_type& B,
                    const typename F::assignment_type& c, const dataset& ds,
                    double smoothing);

  /**
   * Learns a conditional factor P(A | B, C=c) from data,
   * using no regularization.
   * @param ds         Training data.
   */
  template <typename F>
  F
  learn_conditional(const typename F::domain_type& A,
                    const typename F::domain_type& B,
                    const typename F::assignment_type& c, const dataset& ds) {
    return learn_conditional<F>(A, B, c, ds, 0);
  }

  // Specializations of learn_marginal, learn_conditional
  //============================================================================

  /**
   * Learns a marginal table factor P(X) from data.
   * @param X          Variables in marginal.
   * @param ds         Training data.
   * @param smoothing  Regularization (>= 0). This adds smoothing to each
   *                   entry in the learned table factor.
   *                   (default = 0)
   */
  template <>
  table_factor
  learn_marginal<table_factor>(const finite_domain& X, const dataset& ds,
                               double smoothing);

  /**
   * Learns a conditional table factor P(A | B, C=c) from data.
   * @param ds         Training data.
   * @param smoothing  Regularization (>= 0). This adds smoothing to each
   *                   entry in the learned table factor.
   *                   (default = 0)
   */
  template <>
  table_factor
  learn_conditional<table_factor>(const finite_domain& A,
                                  const finite_domain& B,
                                  const finite_assignment& c, const dataset& ds,
                                  double smoothing);

  /**
   * Learns a marginal Gaussian factor P(X) from data.
   * @param X        Variables in marginal.
   * @param ds       Training data.
   * @param reg_cov  Regularization for the covariance matrix.
   *                 This adds a diagonal matrix with reg_cov along the
   *                 diagonal to the empirical covariance matrix.
   *                 (So it is proportional to the number of pseudoexamples.)
   *                 (default = 0)
   */
  template <>
  moment_gaussian
  learn_marginal<moment_gaussian>(const vector_domain& Xdom, const dataset& ds,
                                  double reg_cov);

  /**
   * Learns a marginal Gaussian factor P(X) from data.
   * @param X        Variables in marginal.
   * @param ds       Training data.
   * @param reg_cov  Regularization for the covariance matrix.
   *                 This adds a diagonal matrix with reg_cov along the
   *                 diagonal to the empirical covariance matrix.
   *                 (So it is proportional to the number of pseudoexamples.)
   *                 (default = 0)
   */
  template <>
  canonical_gaussian
  learn_marginal<canonical_gaussian>(const vector_domain& Xdom,
                                     const dataset& ds, double reg_cov);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARN_FACTOR_HPP
