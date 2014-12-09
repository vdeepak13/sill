#ifndef SILL_FACTOR_MLE_HPP
#define SILL_FACTOR_MLE_HPP

#include <sill/factor/factor_mle_incremental.hpp>

#include <stdexcept>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Specifies what to do about missing data.
   *
   * Possible values:
   * - DISALLOW disallows missing data entirely. The results are undefined if
   *   any data is missing.
   * - STRICT allows missing data, but an exception is thrown if the estimates
   *   cannot be computed exactly, e.g. if only some of the head variables are
   *   missing.
   * - SKIP skips over all datapoints that contain any missing data.
   */
  enum missing_data_enum { NO_MISSING, STRICT_MISSING, SKIP_MISSING};

  /**
   * A utility class that represents a maximum-likelihood estimator of
   * the factor distribution. The constructor accepts a pointer to the
   * dataset and, optionally, the estimator parameters. The factor_mle
   * class then acts as functor, accepting the arguments and computing
   * the corresponding marginal or distribution.
   *
   * The default template simply delegates to factor_mle_incremental.
   *
   * \tparam Factor a type that satisfies the distribution and 
   */
  template <typename F>
  class factor_mle {
  public:

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! The argument vector type of the factor
    typedef typename F::var_vector_type var_vector_type;
    
    //! The dataset type used by the factor
    typedef typename F::dataset_type dataset_type;

    //! The record type used by the factor
    typedef typename dataset_type::record_type record_type;

    //! The parameter type of the estimator
    typedef typename factor_mle_incremental<F>::param_type param_type;

    /**
     * Creates a maximum likelihood estimator for a dataset and parameters.
     */
    factor_mle(const dataset_type* ds,
               const param_type& params = param_type(),
               missing_data_enum missing_data = NO_MISSING)
      : ds_(ds), params_(params), missing_data_(missing_data) { }

    /**
     * Returns the marginal distribution over a sequence of variables.
     */
    F operator()(const var_vector_type& args) const {
      factor_mle_incremental<F> estimator(args, params_);
      foreach (const record_type& r, ds_->records(args)) {
        if (missing_data_ == STRICT_MISSING) {
          size_t nmissing = r.count_missing();
          if (nmissing == args.size()) { // none of args are observed, skip
            continue;
          } else if (nmissing) { // cannot handle partially observable args
            throw std::runtime_error(
              "factor_mle: Some (but not all) of the variables are missing"
            );
          }
        } else if (missing_data_ == SKIP_MISSING && r.count_missing()) {
          continue;
        }
        estimator.process(r.values, r.weight);
      }
      return estimator.estimate();
    }

    /**
     * Returns the marginal distribution over a set of variables.
     */
    F operator()(const domain_type& args) const {
      return operator()(make_vector(args));
    }

    /**
     * Returns the conditional distribution p(head | tail).
     */
    F operator()(const var_vector_type& head,
                 const var_vector_type& tail) const {
      factor_mle_incremental<F> estimator(head, tail, params_);
      foreach (const record_type& r, ds_->records(concat(head, tail))) {
        if (missing_data_ == STRICT_MISSING) {
          size_t head_missing = r.count_missing(head);
          size_t tail_missing = r.count_missing(tail);
          if (head_missing == head.size()) { // safe to skip
            continue;
          }
          if (head_missing) { // cannot handle partially observable head
            throw std::runtime_error(
              "factor_mle: Some (but not all) of the head variables are missing"
            );
          }
          if (tail_missing) { // cannot handle unobserved tail
            throw std::runtime_error(
              "factor_mle: Head is fully observed, but the tail is not"
            );
          }
        } else if (missing_data_ == SKIP_MISSING && r.count_missing()) {
          continue;
        }
        estimator.process(r.values, r.weight);
      }
      return estimator.estimate();
    }

    /**
     * Returns the conditional distribution p(head | tail).
     */
    F operator()(const domain_type& head,
                 const domain_type& tail) const {
      return operator()(make_vector(head), make_vector(tail));
    }

  private:
    const dataset_type* ds_;
    param_type params_;
    missing_data_enum missing_data_;

  }; // class factor_mle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
