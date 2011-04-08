
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
   * Class for learning marginal and conditional factors from data.
   *
   * This was built as a struct to allow for template specialization according
   * to factor type but not linear algebra type.
   */
  template <typename F>
  struct learn_factor {

    /**
     * Learns a marginal factor P(X) from data.
     * @param X          Variables in marginal.
     * @param ds         Training data.
     * @param smoothing  Regularization (>= 0).
     */
    template <typename LA>
    static
    F
    learn_marginal(const typename F::domain_type& X, const dataset<LA>& ds,
                   double smoothing);

    /**
     * Learns a marginal factor P(X) from data, using no regularization.
     * @param X          Variables in marginal.
     * @param ds         Training data.
     */
    template <typename LA>
    static
    F
    learn_marginal(const typename F::domain_type& X, const dataset<LA>& ds) {
      return learn_marginal<LA>(X, ds, 0);
    }

    /**
     * Learns a conditional factor P(A | B, C=c) from data.
     * @param ds         Training data.
     * @param smoothing  Regularization (>= 0).
     */
    template <typename LA>
    static
    F
    learn_conditional(const typename F::domain_type& A,
                      const typename F::domain_type& B,
                      const typename F::assignment_type& c,
                      const dataset<LA>& ds,
                      double smoothing);

    /**
     * Learns a conditional factor P(A | B, C=c) from data,
     * using no regularization.
     * @param ds         Training data.
     */
    template <typename LA>
    static
    F
    learn_conditional(const typename F::domain_type& A,
                      const typename F::domain_type& B,
                      const typename F::assignment_type& c,
                      const dataset<LA>& ds) {
      return learn_conditional<LA>(A, B, c, ds, 0);
    }

  }; // struct learn_factor

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
  template <typename LA>
  table_factor
  learn_factor<table_factor>::learn_marginal
  (const finite_domain& X, const dataset<LA>& ds, double smoothing) {
    assert(ds.has_variables(X));
    assert(smoothing >= 0);
    assert(ds.size() > 0);

    table_factor factor(X, smoothing);

    size_t i(0);
    typename dataset<LA>::assignment_iterator end_it(ds.end_assignments());
    for (typename dataset<LA>::assignment_iterator a_it(ds.begin_assignments());
         a_it != end_it; ++a_it) {
      const finite_assignment& a = *a_it;
      factor.v(a) += ds.weight(i);
      ++i;
    }

    factor.normalize();
    return factor;
  }

  /**
   * Learns a conditional table factor P(A | B, C=c) from data.
   * @param ds         Training data.
   * @param smoothing  Regularization (>= 0). This adds smoothing to each
   *                   entry in the learned table factor.
   *                   (default = 0)
   */
  template <>
  template <typename LA>
  table_factor
  learn_factor<table_factor>::learn_conditional
  (const finite_domain& A, const finite_domain& B, const finite_assignment& c,
   const dataset<LA>& ds, double smoothing) {
    assert(ds.has_variables(A));
    assert(ds.has_variables(B));
    finite_domain Cvars(keys(c));
    assert(ds.has_variables(Cvars));
    finite_domain ABvars(set_union(A,B));
    assert(ABvars.size() == A.size() + B.size());
    assert(set_disjoint(ABvars, Cvars));
    assert(smoothing >= 0);
    assert(ds.size() > 0);
    if (A.empty())
      return table_factor(1);

    // Create a constant factor over A,B
    table_factor factor(ABvars, smoothing);

    size_t i(0);
    double total_weight(0);
    typename dataset<LA>::assignment_iterator end_it(ds.end_assignments());
    for (typename dataset<LA>::assignment_iterator a_it(ds.begin_assignments());
         a_it != end_it; ++a_it) {
      const finite_assignment& fa = a_it->finite();
      bool matches_c(true);
      for (finite_assignment::const_iterator c_it(c.begin());
           c_it != c.end(); ++c_it) {
        finite_assignment::const_iterator fa_it(fa.find(c_it->first));
        assert(fa_it != fa.end());
        if (fa_it->second != c_it->second) {
          matches_c = false;
          break;
        }
      }
      total_weight += ds.weight(i);
      if (!matches_c)
        continue;
      factor.v(fa) += ds.weight(i);
      ++i;
    }
    factor.normalize();
    factor = factor.conditional(B);

    return factor;
  } // learn_conditional()

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
  template <typename LA>
  moment_gaussian
  learn_factor<moment_gaussian>::learn_marginal
  (const vector_domain& Xdom, const dataset<LA>& ds, double reg_cov) {
    assert(ds.size() > 0);
    assert(reg_cov >= 0.);
    assert(ds.has_variables(Xdom));
    vector_var_vector X(Xdom.begin(), Xdom.end());
    vec mu;
    ds.mean(mu, X);
    mat cov;
    ds.covariance(cov, X);
    if (reg_cov > 0)
      cov += (reg_cov / ds.size()) * identity(X.size());
    return moment_gaussian(X, mu, cov);
  }

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
  template <typename LA>
  canonical_gaussian
  learn_factor<canonical_gaussian>::learn_marginal
  (const vector_domain& Xdom, const dataset<LA>& ds, double reg_cov) {
    assert(ds.size() > 0);
    assert(reg_cov >= 0.);
    assert(ds.has_variables(Xdom));
    vector_var_vector X(Xdom.begin(), Xdom.end());
    vec mu;
    mat cov;
    ds.mean_covariance(mu, cov, X);
    if (reg_cov > 0)
      cov += (reg_cov / ds.size()) * identity(X.size());
    mat lambda;
    bool result = inv(cov, lambda);
    if (!result)
      throw std::runtime_error
        ("Matrix inverse failed in canonical_gaussian::learn_marginal().");
    return canonical_gaussian(X, lambda, lambda * mu);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARN_FACTOR_HPP
