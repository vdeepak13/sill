#ifndef SILL_FACTOR_MLE_HPP
#define SILL_FACTOR_MLE_HPP

#include <sill/factor/factor_mle_incremental.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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
               const param_type& params = param_type())
      : ds_(ds), params_(params) { }

    /**
     * Returns the marginal distribution over a sequence of variables.
     */
    F operator()(const var_vector_type& args) const {
      factor_mle_incremental<F> estimator(args, params_);
      foreach (const record_type& r, ds_->records(args)) {
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
      foreach (const record_type& r, ds_->records(concat(tail, head))) {
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

  }; // class factor_mle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
