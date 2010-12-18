#ifndef SILL_LEARN_CRF_FACTOR_HPP
#define SILL_LEARN_CRF_FACTOR_HPP

#include <sill/base/variables.hpp>
#include <sill/factor/hybrid_crf_factor.hpp>
#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/log_reg_crf_factor.hpp>
#include <sill/factor/table_crf_factor.hpp>
#include <sill/learning/crossval_methods.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/learn_factor.hpp>
#include <sill/math/constants.hpp>
#include <sill/math/free_functions.hpp>

#include <sill/macros_def.hpp>

/**
 * \file learn_crf_factors.hpp  Methods for learning CRF factors from data.
 *
 * File contents:
 *  - Declarations: Learning CRF Factors with Fixed Regularization
 *  - Declarations: Learning CRF Factors with Regularization Chosen via CV
 *  - Definitions: Learning CRF Factors with Fixed Regularization
 *  - Definitions: Learning CRF Factors with Regularization Chosen via CV
 */

// Macro for specializing learn_crf_factor to hybrid_crf_factor instances.
// (Specialization declaration)
#define GEN_LEARN_CRF_FACTOR_HYBRID_DECL(F)                             \
  template <>                                                           \
  hybrid_crf_factor<F>*                                                 \
  learn_crf_factor<hybrid_crf_factor<F> >                               \
  (boost::shared_ptr<dataset> ds_ptr,                                   \
   const hybrid_crf_factor<F>::output_domain_type& Y_,                  \
   copy_ptr<hybrid_crf_factor<F>::input_domain_type> X_ptr_,            \
   const hybrid_crf_factor<F>::parameters& params,                      \
   unsigned random_seed);

namespace sill {

  // Declarations: Learning CRF Factors with Fixed Regularization
  //============================================================================

  /**
   * Returns a newly allocated factor which represents P(Y, X) or P(Y | X),
   * learned from data;
   * the main guarantee is that conditioning the factor on X=x AND normalizing
   * it will produce an estimate of P(Y | X=x).
   *
   * @param ds_ptr   Training data.
   * @param Y        Output variables Y.
   * @param X_ptr    Input variables X.
   * @param params   CRF factor parameters used for training options.
   *
   * @tparam F  CRF factor type
   */
  template <typename F>
  F*
  learn_crf_factor(boost::shared_ptr<dataset> ds_ptr,
                   const typename F::output_domain_type& Y_,
                   copy_ptr<typename F::input_domain_type> X_ptr_,
                   const typename F::parameters& params,
                   unsigned random_seed = time(NULL));

  //! Specialization: table_crf_factor
  template <>
  table_crf_factor*
  learn_crf_factor<table_crf_factor>(boost::shared_ptr<dataset> ds_ptr,
                                     const finite_domain& Y_,
                                     copy_ptr<finite_domain> X_ptr_,
                                     const table_crf_factor::parameters& params,
                                     unsigned random_seed);

  //! Specialization: log_reg_crf_factor
  template <>
  log_reg_crf_factor*
  learn_crf_factor<log_reg_crf_factor>
  (boost::shared_ptr<dataset> ds_ptr,
   const finite_domain& Y_, copy_ptr<domain> X_ptr_,
   const log_reg_crf_factor::parameters& params, unsigned random_seed);

  //! Specialization: gaussian_crf_factor
  template <>
  gaussian_crf_factor*
  learn_crf_factor<gaussian_crf_factor>
  (boost::shared_ptr<dataset> ds_ptr,
   const vector_domain& Y_, copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed);

  //! Specialization: hybrid_crf_factor<gaussian_crf_factor>
  GEN_LEARN_CRF_FACTOR_HYBRID_DECL(gaussian_crf_factor)

  // Declarations: Learning CRF Factors with Regularization Chosen via CV
  //============================================================================

  /**
   * Returns a newly allocated factor which represents P(Y, X) or P(Y | X),
   * learned from data;
   * the main guarantee is that conditioning the factor on X=x AND normalizing
   * it will produce an estimate of P(Y | X=x).
   *
   * This does cross validation to choose regularization.
   * (See crossval_parameters for more info.)
   *
   * This is a generic version of learn_crf_factor_cv();
   * for some types of factors, it is more efficient to do a specialized
   * implementation.
   * This relies on the factor having an implementation of learn_crf_factor().
   *
   * @param reg_params (Return value.) Parameters which were tried.
   * @param means      (Return value.) Means of scores for the given lambdas.
   * @param stderrs    (Return value.) Std errors of scores for the lambdas.
   * @param cv_params  Parameters specifying how to do cross validation.
   */
  template <typename F>
  F*
  learn_crf_factor_cv
  (std::vector<typename F::regularization_type>& reg_params,
   vec& means, vec& stderrs,
   const crossval_parameters<F::regularization_type::nlambdas>& cv_params,
   boost::shared_ptr<dataset> ds_ptr,
   const typename F::output_domain_type& Y_,
   copy_ptr<typename F::input_domain_type> X_ptr_,
   const typename F::parameters& params,
   unsigned random_seed = time(NULL));

  //! Specialization: gaussian_crf_factor
  template <>
  gaussian_crf_factor*
  learn_crf_factor_cv<gaussian_crf_factor>
  (std::vector<gaussian_crf_factor::regularization_type>& reg_params,
   vec& means, vec& stderrs,
   const
   crossval_parameters<gaussian_crf_factor::regularization_type::nlambdas>&
   cv_params,
   boost::shared_ptr<dataset> ds_ptr, const vector_domain& Y_,
   copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed);

  namespace impl {

    /**
     * CrossvalFunctor used by the generic learn_crf_factor_cv() below
     * when it calls crossval_zoom().
     *
     * @see learn_crf_factor_cv, crossval_zoom
     */
    template <typename F>
    class learn_crf_factor_cv_functor;

    /**
     * CrossvalFunctor used by learn_crf_factor_cv<gaussian_crf_factor>() below
     * when it calls crossval_zoom().
     *
     * @see learn_crf_factor_cv, crossval_zoom
     */
    class gcf_learn_crf_factor_cv_functor;

  }; // namespace impl

  // Definitions: Learning CRF Factors with Regularization Chosen via CV
  //============================================================================

  namespace impl {

    template <typename F>
    class learn_crf_factor_cv_functor {

      boost::shared_ptr<dataset> ds_ptr;

      const typename F::output_domain_type* Y_ptr;

      copy_ptr<typename F::input_domain_type> X_ptr_;

      const typename F::parameters* params_ptr;

    public:

      //! Constructor.
      learn_crf_factor_cv_functor(boost::shared_ptr<dataset> ds_ptr,
                                 const typename F::output_domain_type& Y_,
                                 copy_ptr<typename F::input_domain_type> X_ptr_,
                                 const typename F::parameters& params)
        : ds_ptr(ds_ptr), Y_ptr(&Y_), X_ptr_(X_ptr_), params_ptr(&params) { }

      //! Try the given lambdas, and returns means,stderrs of results.
      vec operator()(vec& means, vec& stderrs, const std::vector<vec>& lambdas,
                     size_t nfolds, unsigned random_seed) const {

        assert(params_ptr->valid());
        assert(nfolds <= ds_ptr->size());

        means.resize(lambdas.size());
        means.zeros_memset();
        stderrs.resize(lambdas.size());
        stderrs.zeros_memset();
        boost::mt11213b rng(random_seed);
        boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
        dataset_view permuted_view(*ds_ptr);
        permuted_view.set_record_indices(randperm(ds_ptr->size(), rng));
        typename F::parameters tmp_params(*params_ptr);
        boost::shared_ptr<dataset_view>
          fold_train_view_ptr(new dataset_view(permuted_view));
        dataset_view fold_test_view(permuted_view);
        fold_train_view_ptr->save_record_view();
        fold_test_view.save_record_view();
        // For each fold
        for (size_t fold(0); fold < nfolds; ++fold) {
          // Prepare the fold dataset views
          if (fold != 0) {
            fold_train_view_ptr->restore_record_view();
            fold_test_view.restore_record_view();
          }
          fold_train_view_ptr->set_cross_validation_fold(fold, nfolds, false);
          fold_test_view.set_cross_validation_fold(fold, nfolds, true);
          for (size_t k(0); k < lambdas.size(); ++k) {
            if (!is_finite(means[k]))
              continue;
            tmp_params.reg.lambdas = lambdas[k];
            try {
              F* tmpf = learn_crf_factor<F>(fold_train_view_ptr, *Y_ptr, X_ptr_,
                                           tmp_params, unif_int(rng));
              double tmpval(tmpf->log_expected_value(fold_test_view));
              delete(tmpf);
              if (is_finite(means[k])) {
                means[k] -= tmpval;
                stderrs[k] += tmpval * tmpval;
              }
            } catch(const std::runtime_error& e) {
              // Assume the regularization must be stronger.
              means[k] = std::numeric_limits<double>::infinity();
              stderrs[k] = std::numeric_limits<double>::infinity();
            }
          }
        }
        for (size_t k(0); k < means.size(); ++k) {
          if (is_finite(means[k])) {
            means[k] /= nfolds;
            stderrs[k] = std::sqrt((stderrs[k] / nfolds) - (means[k]*means[k]));
          }
        }
        return lambdas[max_index(means, rng)];
      } // operator()

    }; // class learn_crf_factor_cv_functor

    class gcf_learn_crf_factor_cv_functor {

      boost::shared_ptr<dataset> ds_ptr;

      const vector_domain* Y_ptr;

      copy_ptr<vector_domain> X_ptr;

      const gaussian_crf_factor::parameters* params_ptr;

    public:

      //! Constructor.
      gcf_learn_crf_factor_cv_functor
      (boost::shared_ptr<dataset> ds_ptr,
       const vector_domain& Y_, copy_ptr<vector_domain> X_ptr,
       const gaussian_crf_factor::parameters& params);

      /**
       * Try the given lambdas, and returns means,stderrs of results.
       */
      vec operator()(vec& means, vec& stderrs, const std::vector<vec>& lambdas,
                     size_t nfolds, unsigned random_seed) const;

    }; // class gcf_learn_crf_factor_cv_functor

  }; // namespace impl

  template <typename F>
  F*
  learn_crf_factor_cv
  (std::vector<typename F::regularization_type>& reg_params,
   vec& means, vec& stderrs,
   const crossval_parameters<F::regularization_type::nlambdas>& cv_params,
   boost::shared_ptr<dataset> ds_ptr,
   const typename F::output_domain_type& Y_,
   copy_ptr<typename F::input_domain_type> X_ptr_,
   const typename F::parameters& params,
   unsigned random_seed = time(NULL)) {

    assert(params.valid());
    assert(cv_params.valid());

    boost::mt11213b rng(random_seed);
    boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());

    std::vector<vec> lambdas;
    impl::learn_crf_factor_cv_functor<F> cvfunctor(ds_ptr, Y_, X_ptr_, params);
    vec best_lambda =
      crossval_zoom
      <impl::learn_crf_factor_cv_functor<F>, F::regularization_type::nlambdas>
      (lambdas, means, stderrs, cv_params, cvfunctor, unif_int(rng));
    assert(best_lambda.size() == F::regularization_type::nlambdas);

    reg_params.clear();
    typename F::regularization_type reg;
    reg.regularization = params.reg.regularization;
    foreach(const vec& v, lambdas) {
      reg.lambdas = v;
      reg_params.push_back(reg);
    }
    typename F::parameters tmp_params(params);
    tmp_params.reg.lambdas = best_lambda;
    return learn_crf_factor<F>(ds_ptr, Y_, X_ptr_, tmp_params, unif_int(rng));

  } // learn_crf_factor_cv()

  //! Specialization: gaussian_crf_factor
  template <>
  gaussian_crf_factor*
  learn_crf_factor_cv<gaussian_crf_factor>
  (std::vector<gaussian_crf_factor::regularization_type>& reg_params,
   vec& means, vec& stderrs,
   const
   crossval_parameters<gaussian_crf_factor::regularization_type::nlambdas>&
   cv_params,
   boost::shared_ptr<dataset> ds_ptr, const vector_domain& Y_,
   copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed);

};  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LEARN_CRF_FACTOR_HPP
