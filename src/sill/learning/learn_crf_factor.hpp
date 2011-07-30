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
#include <sill/math/permutations.hpp>

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
  hybrid_crf_factor<F>                                                  \
  learn_crf_factor<hybrid_crf_factor<F> >::train                        \
  (const dataset<F::la_type>& ds,                                       \
   const hybrid_crf_factor<F>::output_domain_type& Y_,                  \
   copy_ptr<hybrid_crf_factor<F>::input_domain_type> X_ptr_,            \
   const hybrid_crf_factor<F>::parameters& params,                      \
   unsigned random_seed);

namespace sill {

  //============================================================================
  // Learning generic factors
  //============================================================================

  // Forward declaration
  namespace impl {
    template <typename F>
    F
    learn_crf_factor_cv_generic
    (std::vector<typename F::regularization_type>& reg_params,
     vec& means, vec& stderrs,
     const crossval_parameters& cv_params,
     const dataset<typename F::la_type>& ds,
     const typename F::output_domain_type& Y_,
     copy_ptr<typename F::input_domain_type> X_ptr_,
     const typename F::parameters& params,
     unsigned random_seed);
  }

  /**
   * Struct for learning CRF factors from data.
   * This is a struct to permit partial specialization for, e.g.,
   * supporting factors which can use multiple linear algebra types.
   */
  template <typename F>
  struct learn_crf_factor {

    /**
     * Returns a newly allocated factor which represents P(Y, X) or P(Y | X),
     * learned from data;
     * the main guarantee is that conditioning the factor on X=x AND normalizing
     * it will produce an estimate of P(Y | X=x).
     *
     * @param ds       Training data.
     * @param Y        Output variables Y.
     * @param X_ptr    Input variables X.
     * @param params   CRF factor parameters used for training options.
     *
     * @tparam F  CRF factor type
     */
    static F
    train(const dataset<typename F::la_type>& ds,
          const typename F::output_domain_type& Y_,
          copy_ptr<typename F::input_domain_type> X_ptr_,
          const typename F::parameters& params,
          unsigned random_seed = time(NULL));

    //! @deprecated
    static F
    train_cv
    (std::vector<typename F::regularization_type>& reg_params,
     vec& means, vec& stderrs,
     const crossval_parameters& cv_params,
     const dataset<typename F::la_type>& ds,
     const typename F::output_domain_type& Y_,
     copy_ptr<typename F::input_domain_type> X_ptr_,
     const typename F::parameters& params,
     unsigned random_seed = time(NULL)) {
      return impl::learn_crf_factor_cv_generic<F>
        (reg_params, means, stderrs,
         cv_params, ds, Y_, X_ptr_, params, random_seed);
    }

    /**
     * Returns a newly allocated factor which represents P(Y, X) or P(Y | X),
     * learned from data;
     * the main guarantee is that conditioning the factor on X=x AND normalizing
     * it will produce an estimate of P(Y | X=x).
     *
     * This does cross validation to choose regularization.
     * (See crossval_parameters for more info.)
     *
     * This is a generic version of learn_crf_factor::train_cv;
     * for some types of factors, it is more efficient to do a specialized
     * implementation.
     * This relies on an implementation of learn_crf_factor::train.
     *
     * @param reg_params (Return value.) Parameters which were tried.
     * @param means      (Return value.) Means of scores for the given lambdas.
     * @param stderrs    (Return value.) Std errors of scores for the lambdas.
     * @param cv_params  Parameters specifying how to do cross validation.
     */
    static F
    train_cv
    (const crossval_parameters& cv_params,
     const dataset<typename F::la_type>& ds,
     const typename F::output_domain_type& Y_,
     copy_ptr<typename F::input_domain_type> X_ptr_,
     const typename F::parameters& params,
     unsigned random_seed = time(NULL)) {
      std::vector<typename F::regularization_type> reg_params;
      vec means;
      vec stderrs;
      return train_cv(reg_params, means, stderrs,
                      cv_params, ds, Y_, X_ptr_, params, random_seed);
    }

  }; // struct learn_crf_factor

  //============================================================================
  // Specialization: table_crf_factor
  //============================================================================

  template <>
  table_crf_factor
  learn_crf_factor<table_crf_factor>::train
  (const dataset<table_crf_factor::la_type>& ds,
   const finite_domain& Y_,
   copy_ptr<finite_domain> X_ptr_,
   const table_crf_factor::parameters& params,
   unsigned random_seed);

  //============================================================================
  // Specialization: log_reg_crf_factor
  //============================================================================

  template <typename LA>
  struct learn_crf_factor<log_reg_crf_factor<LA> > {

    static log_reg_crf_factor<LA>
    train
    (const dataset<LA>& ds,
     const finite_domain& Y_, copy_ptr<domain> X_ptr_,
     const typename log_reg_crf_factor<LA>::parameters& params,
     unsigned random_seed);

    //! @deprecated
    static log_reg_crf_factor<LA>
    train_cv
    (std::vector<typename log_reg_crf_factor<LA>::regularization_type>&
     reg_params,
     vec& means, vec& stderrs,
     const crossval_parameters& cv_params,
     const dataset<typename log_reg_crf_factor<LA>::la_type>& ds,
     const finite_domain& Y_, copy_ptr<domain> X_ptr_,
     const typename log_reg_crf_factor<LA>::parameters& params,
     unsigned random_seed) {
      return impl::learn_crf_factor_cv_generic<log_reg_crf_factor<LA> >
        (reg_params, means, stderrs,
         cv_params, ds, Y_, X_ptr_, params, random_seed);
    }

    static log_reg_crf_factor<LA>
    train_cv
    (const crossval_parameters& cv_params,
     const dataset<typename log_reg_crf_factor<LA>::la_type>& ds,
     const finite_domain& Y_, copy_ptr<domain> X_ptr_,
     const typename log_reg_crf_factor<LA>::parameters& params,
     unsigned random_seed) {
      std::vector<typename log_reg_crf_factor<LA>::regularization_type>
        reg_params;
      vec means;
      vec stderrs;
      return train_cv(reg_params, means, stderrs,
                      cv_params, ds, Y_, X_ptr_, params, random_seed);
    }

  };

  //============================================================================
  // Specialization: gaussian_crf_factor
  //============================================================================

  template <>
  gaussian_crf_factor
  learn_crf_factor<gaussian_crf_factor>::train
  (const dataset<gaussian_crf_factor::la_type>& ds,
   const vector_domain& Y_, copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed);

  template <>
  gaussian_crf_factor
  learn_crf_factor<gaussian_crf_factor>::train_cv
  (std::vector<gaussian_crf_factor::regularization_type>& reg_params,
   vec& means, vec& stderrs,
   const crossval_parameters& cv_params,
   const dataset<gaussian_crf_factor::la_type>& ds, const vector_domain& Y_,
   copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed);

  template <>
  gaussian_crf_factor
  learn_crf_factor<gaussian_crf_factor>::train_cv
  (const crossval_parameters& cv_params,
   const dataset<gaussian_crf_factor::la_type>& ds, const vector_domain& Y_,
   copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed);

  //============================================================================
  // Specialization: hybrid_crf_factor<gaussian_crf_factor>
  //============================================================================

  GEN_LEARN_CRF_FACTOR_HYBRID_DECL(gaussian_crf_factor)


  //============================================================================
  // Implementations
  //============================================================================


  namespace impl {

    //! Functor for default train_cv implementation
    template <typename F>
    class learn_crf_factor_cv_functor {

      const dataset<typename F::la_type>& ds;

      const typename F::output_domain_type* Y_ptr;

      copy_ptr<typename F::input_domain_type> X_ptr_;

      const typename F::parameters* params_ptr;

    public:

      //! Constructor.
      learn_crf_factor_cv_functor(const dataset<typename F::la_type>& ds,
                                 const typename F::output_domain_type& Y_,
                                 copy_ptr<typename F::input_domain_type> X_ptr_,
                                 const typename F::parameters& params)
        : ds(ds), Y_ptr(&Y_), X_ptr_(X_ptr_), params_ptr(&params) { }

      //! Try the given lambdas, and returns means,stderrs of results.
      vec operator()(vec& means, vec& stderrs, const std::vector<vec>& lambdas,
                     size_t nfolds, unsigned random_seed) const {

        assert(params_ptr->valid());
        assert(nfolds <= ds.size());

        means.zeros(lambdas.size());
        stderrs.zeros(lambdas.size());
        boost::mt11213b rng(random_seed);
        boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
        dataset_view<typename F::la_type> permuted_view(ds);
        permuted_view.set_record_indices(randperm(ds.size(), rng));
        typename F::parameters tmp_params(*params_ptr);
        dataset_view<typename F::la_type> fold_train_view(permuted_view);
        dataset_view<typename F::la_type> fold_test_view(permuted_view);
        fold_train_view.save_record_view();
        fold_test_view.save_record_view();
        // For each fold
        for (size_t fold(0); fold < nfolds; ++fold) {
          // Prepare the fold dataset views
          if (fold != 0) {
            fold_train_view.restore_record_view();
            fold_test_view.restore_record_view();
          }
          fold_train_view.set_cross_validation_fold(fold, nfolds, false);
          fold_test_view.set_cross_validation_fold(fold, nfolds, true);
          for (size_t k(0); k < lambdas.size(); ++k) {
            if (!is_finite(means[k]))
              continue;
            tmp_params.reg.lambdas = lambdas[k];
            try {
              F tmpf =
                learn_crf_factor<F>::train(fold_train_view, *Y_ptr, X_ptr_,
                                           tmp_params, unif_int(rng));
              double tmpval(tmpf.log_expected_value(fold_test_view));
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

    //! Default train_cv implementation
    template <typename F>
    F
    learn_crf_factor_cv_generic
    (std::vector<typename F::regularization_type>& reg_params,
     vec& means, vec& stderrs,
     const crossval_parameters& cv_params,
     const dataset<typename F::la_type>& ds,
     const typename F::output_domain_type& Y_,
     copy_ptr<typename F::input_domain_type> X_ptr_,
     const typename F::parameters& params,
     unsigned random_seed) {

      assert(params.valid());
      assert(cv_params.valid());

      boost::mt11213b rng(random_seed);
      boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());

      std::vector<vec> lambdas;
      impl::learn_crf_factor_cv_functor<F> cvfunctor(ds, Y_, X_ptr_, params);
      vec best_lambda =
        crossval_zoom<impl::learn_crf_factor_cv_functor<F> >
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
      return
        learn_crf_factor<F>::train(ds, Y_, X_ptr_, tmp_params, unif_int(rng));
    } // learn_crf_factor_cv_generic

  } // namespace impl


  //! Specialization: log_reg_crf_factor
  template <typename LA>
  log_reg_crf_factor<LA>
  learn_crf_factor<log_reg_crf_factor<LA> >::train
  (const dataset<LA>& ds,
   const finite_domain& Y_, copy_ptr<domain> X_ptr_,
   const typename log_reg_crf_factor<LA>::parameters& params,
   unsigned random_seed) {

    assert(Y_.size() != 0);
    if (!ds.has_variables(Y_)) {
      std::cerr << "learn_crf_factor given Y_ = " << Y_
                << ", but the dataset only contains finite variables: "
                << finite_domain(ds.finite_variables().first,
                                 ds.finite_variables().second)
                << std::endl;
      assert(false);
    }
    assert(X_ptr_);

    assert(ds.has_variables(*X_ptr_));
    assert(params.valid());

    if (Y_.size() == 1) {
      dataset_statistics<LA> stats(ds);
      // Train multiclass logistic regressor
      multiclass_logistic_regression_parameters mlr_params(params.mlr_params);
      mlr_params.regularization = params.reg.regularization;
      mlr_params.lambda = params.reg.lambdas[0];
      mlr_params.random_seed = random_seed;
      boost::shared_ptr<multiclass_logistic_regression<LA> >
        mlr_ptr(new multiclass_logistic_regression<LA>(ds, mlr_params));
      return
        log_reg_crf_factor<LA>
        (boost::shared_ptr<multiclass2multilabel<LA> >
         (new multiclass2multilabel<LA>(mlr_ptr, ds)),
         params.smoothing / ds.size(), Y_, X_ptr_);
    } else { // then Y_.size() > 1
      // Set up dataset view
      dataset_view<LA> ds_view(ds);
      std::set<size_t> finite_indices;
      size_t new_class_var_size = num_assignments(Y_);
      foreach(finite_variable* v, Y_) {
        finite_indices.insert(ds_view.record_index(v));
      }
      std::set<size_t> vector_indices;
      foreach(variable* v, *X_ptr_) {
        switch(v->get_variable_type()) {
        case variable::FINITE_VARIABLE:
          finite_indices.insert
            (ds_view.record_index(dynamic_cast<finite_variable*>(v)));
          break;
        case variable::VECTOR_VARIABLE:
          vector_indices.insert
            (ds_view.record_index(dynamic_cast<vector_variable*>(v)));
          break;
        default:
          assert(false);
        }
      }
      ds_view.set_variable_indices(finite_indices, vector_indices);
      ds_view.set_finite_class_variables(Y_);
      dataset_statistics<LA> stats(ds_view);
      // Train multilabel logistic regressor
      multiclass_logistic_regression_parameters mlr_params(params.mlr_params);
      mlr_params.regularization = params.reg.regularization;
      mlr_params.lambda = params.reg.lambdas[0];
      finite_variable* new_merged_var
        = params.u.new_finite_variable(new_class_var_size);
      multiclass2multilabel_parameters<LA> m2m_params;
      m2m_params.base_learner =
        boost::shared_ptr<multiclass_classifier<LA> >
        (new multiclass_logistic_regression<LA>(mlr_params));
      m2m_params.random_seed = random_seed;
      m2m_params.new_label = new_merged_var;
      return
        log_reg_crf_factor<LA>
        (boost::shared_ptr<multiclass2multilabel<LA> >
         (new multiclass2multilabel<LA>(stats, m2m_params)),
         params.smoothing / ds.size(), Y_, X_ptr_);
    }
  } // learn_crf_factor<log_reg_crf_factor<LA> >::train

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LEARN_CRF_FACTOR_HPP
