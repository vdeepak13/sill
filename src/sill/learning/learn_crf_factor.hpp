#ifndef SILL_LEARN_CRF_FACTOR_HPP
#define SILL_LEARN_CRF_FACTOR_HPP

#include <sill/base/variables.hpp>
#include <sill/factor/hybrid_crf_factor.hpp>
#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/log_reg_crf_factor.hpp>
#include <sill/factor/table_crf_factor.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/learn_factor.hpp>
#include <sill/learning/validation/model_validation_functor.hpp>
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
    (const crossval_parameters& cv_params,
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
      return impl::learn_crf_factor_cv_generic<F>
        (cv_params, ds, Y_, X_ptr_, params, random_seed);
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
    (const crossval_parameters& cv_params,
     const dataset<typename log_reg_crf_factor<LA>::la_type>& ds,
     const finite_domain& Y_, copy_ptr<domain> X_ptr_,
     const typename log_reg_crf_factor<LA>::parameters& params,
     unsigned random_seed) {
      return impl::learn_crf_factor_cv_generic<log_reg_crf_factor<LA> >
        (cv_params, ds, Y_, X_ptr_, params, random_seed);
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
    class learn_crf_factor_val_functor
      : public model_validation_functor<typename F::la_type> {

      // Public types
      //------------------------------------------------------------------------
    public:

      typedef typename F::la_type la_type;

      typedef model_validation_functor<la_type> base;

      typedef typename la_type::value_type value_type;
      typedef arma::Col<value_type>        dense_vector_type;

      // Public methods
      //------------------------------------------------------------------------
    public:

      learn_crf_factor_val_functor()
        : params_ptr(NULL) { }

      //! Constructor. (no CV)
      learn_crf_factor_val_functor
      (const typename F::parameters& params,
       const typename F::output_domain_type& Y_,
       copy_ptr<typename F::input_domain_type> X_ptr_)
        : params_ptr(new typename F::parameters(params)),
          Y_(Y_), X_ptr_(X_ptr_), do_cv(false) {
        assert(X_ptr_);
      }

      //! Constructor. (with CV)
      learn_crf_factor_val_functor
      (const typename F::parameters& params,
       const typename F::output_domain_type& Y_,
       copy_ptr<typename F::input_domain_type> X_ptr_,
       const crossval_parameters& cv_params)
        : params_ptr(new typename F::parameters(params)),
          Y_(Y_), X_ptr_(X_ptr_), do_cv(true), cv_params(cv_params) {
        assert(X_ptr_);
        assert(params_ptr->valid());
        assert(cv_params.valid());
      }

      ~learn_crf_factor_val_functor() {
        if (params_ptr)
          delete(params_ptr);
      }

      learn_crf_factor_val_functor(const learn_crf_factor_val_functor& other) {
        *this = other;
      }

      learn_crf_factor_val_functor&
      operator=(const learn_crf_factor_val_functor& other) {
        if (params_ptr)
          delete(params_ptr);
        params_ptr = new typename F::parameters(*other.params_ptr);
        Y_ = other.Y_;
        X_ptr_ = other.X_ptr_;
        f = other.f;
        do_cv = other.do_cv;
        cv_params = other.cv_params;
      }

      // Protected data
      //------------------------------------------------------------------------
    protected:

      using base::result_map_;

      //! Owned by this class.
      typename F::parameters* params_ptr;

      typename F::output_domain_type Y_;

      copy_ptr<typename F::input_domain_type> X_ptr_;

      F f;

      bool do_cv;

      crossval_parameters cv_params;

      // Protected methods
      //------------------------------------------------------------------------

      void train_model(const dataset<la_type>& ds, unsigned random_seed) {
        assert(params_ptr);
        if (do_cv) {
          boost::mt11213b rng(random_seed);
          boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());

          learn_crf_factor_val_functor lcf_val_func(*params_ptr, Y_, X_ptr_);
          validation_framework<la_type>
            val_frame(ds, cv_params, lcf_val_func, unif_int(rng));
          assert(val_frame.best_lambdas().size()
                 == params_ptr->reg.lambdas.size());
          params_ptr->reg.lambdas = val_frame.best_lambdas();
          random_seed = unif_int(rng);
        }

        f = learn_crf_factor<F>::train(ds, Y_, X_ptr_, *params_ptr,random_seed);
      }

      void train_model(const dataset<la_type>& ds,
                       const dense_vector_type& validation_params,
                       unsigned random_seed) {
        assert(!do_cv);//This would mean choosing lambda within choosing lambda.
        assert(validation_params.size() == params_ptr->reg.lambdas.size());
        params_ptr->reg.lambdas = validation_params;
        train_model(ds, random_seed);
      }

      //! Compute results from model, and store them in result_map_.
      //! @param prefix  Prefix to add to result names.
      //! @return  Main result/score.
      value_type
      add_results(const dataset<la_type>& ds, const std::string& prefix) {
        value_type ll = ds.expected_value(f.log_likelihood());
        result_map_[prefix + "log likelihood"] = ll;
        return ll;
      }

    }; // class learn_crf_factor_val_functor


    //! Default train_cv implementation
    template <typename F>
    F
    learn_crf_factor_cv_generic
    (const crossval_parameters& cv_params,
     const dataset<typename F::la_type>& ds,
     const typename F::output_domain_type& Y_,
     copy_ptr<typename F::input_domain_type> X_ptr_,
     const typename F::parameters& params,
     unsigned random_seed) {

      assert(params.valid());
      assert(cv_params.valid());

      boost::mt11213b rng(random_seed);
      boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());

      learn_crf_factor_val_functor<F> lcf_val_func(params, Y_, X_ptr_);
      validation_framework<typename F::la_type>
        val_frame(ds, cv_params, lcf_val_func, unif_int(rng));
      assert(val_frame.best_lambdas().size()
             == params.reg.lambdas.size());
      typename F::parameters tmp_params(params);
      tmp_params.reg.lambdas = val_frame.best_lambdas();

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

    // Set up dataset view
    dataset_view<LA> ds_view(ds);
    std::set<size_t> finite_indices;
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

    // Train multiclass logistic regressor
    multiclass_logistic_regression_parameters mlr_params(params.mlr_params);
    mlr_params.regularization = params.reg.regularization;
    mlr_params.lambda = params.reg.lambdas[0];
    if (Y_.size() == 1) {
      mlr_params.random_seed = random_seed;
      boost::shared_ptr<multiclass_logistic_regression<LA> >
        mlr_ptr(new multiclass_logistic_regression<LA>(ds_view, mlr_params));
      mlr_ptr->finish_learning();
      return
        log_reg_crf_factor<LA>
        (boost::shared_ptr<multiclass2multilabel<LA> >
         (new multiclass2multilabel<LA>(mlr_ptr, ds_view)),
         params.smoothing / ds.size(), Y_, X_ptr_);
    } else { // then Y_.size() > 1
      dataset_statistics<LA> stats(ds_view);
      finite_variable* new_merged_var
        = params.u.new_finite_variable(num_assignments(Y_));
      multiclass2multilabel_parameters<LA> m2m_params;
      m2m_params.base_learner =
        boost::shared_ptr<multiclass_classifier<LA> >
        (new multiclass_logistic_regression<LA>(mlr_params));
      m2m_params.random_seed = random_seed;
      m2m_params.new_label = new_merged_var;
      boost::shared_ptr<multiclass2multilabel<LA> >
        m2m_ptr(new multiclass2multilabel<LA>(stats, m2m_params));
      m2m_ptr->finish_learning();
      return log_reg_crf_factor<LA>(m2m_ptr, params.smoothing / ds.size(),
                                    Y_, X_ptr_);
    }
  } // learn_crf_factor<log_reg_crf_factor<LA> >::train

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LEARN_CRF_FACTOR_HPP
