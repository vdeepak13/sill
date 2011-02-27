#ifndef SILL_CRF_VALIDATION_FUNCTOR_HPP
#define SILL_CRF_VALIDATION_FUNCTOR_HPP

#include <sill/learning/validation/model_validation_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Model validation functor for Conditional Random Fields (CRFs).
   *
   * @tparam F  CRF factor type.
   */
  template <typename F>
  class crf_validation_functor
    : public model_validation_functor {

    // Public types
    // =========================================================================
  public:

    typedef model_validation_functor base;

    //! Type of CRF factor
    typedef F crf_factor_type;

    //! Type of crf_graph.
    typedef typename crf_model<F>::crf_graph_type crf_graph_type;

    // Constructors and destructors
    // =========================================================================

    /**
     * Constructor which uses a crf_graph.
     */
    crf_validation_functor(const crf_graph_type& structure,
                           const crf_parameter_learner_parameters& cpl_params)
      : structure(structure), cpl_params(cpl_params), use_weights(false) {
    }

    /**
     * Constructor which uses a crf_model.
     *
     * @param use_weights   If true, then use the given model's weights
     *                      (parameters) to initialize learning.
     */
    crf_validation_functor(const crf_model<F>& model,
                           const crf_parameter_learner_parameters& cpl_params,
                           bool use_weights = true)
      : structure(model), model(model), cpl_params(cpl_params),
        use_weights(use_weights) {
    }

    // Public methods
    // =========================================================================

    /**
     * Train a model, and compute results on the test dataset.
     * This version of train() is used for choosing parameters.
     *
     * This method can set the result map if needed,
     * but only the returned value is required by validation_framework.
     *
     * @param train_ds
     *          Training dataset.
     * @param test_ds
     *          Test dataset.
     * @param validation_params 
     *          Used for choosing parameters via the validation_framework.
     * @param warm_start_recommended
     *          If true, then the validation_framework recommends that training
     *          be initialized using the most recently trained model.
     * @param random_seed
     *
     * @return  Result/score.
     */
    double
    train(const dataset& train_ds, const dataset& test_ds,
          const vec& validation_params, bool warm_start_recommended,
          unsigned random_seed = time(NULL)) {
      cpl_params.lambdas = validation_params;
      use_weights = warm_start_recommended;
      train_model(train_ds, random_seed);
      result_map_.clear();
      add_results(train_ds, "train ");
      return add_results(test_ds, "test ");
    } // train (for choosing parameters)

    /**
     * Train a model, and compute results on the test dataset.
     * This version of train() is used for testing (not for choosing
     * parameters).
     *
     * @param train_ds
     *          Training dataset.
     * @param test_ds
     *          Test dataset.
     * @param random_seed
     *
     * @return  Result/score.
     */
    double
    train(const dataset& train_ds, const dataset& test_ds,
          unsigned random_seed = time(NULL)) {
      train_model(train_ds, random_seed);
      result_map_.clear();
      add_results(train_ds, "train ");
      return add_results(test_ds, "test ");
    } // train (for testing)

    /**
     * Compute results on the test dataset.
     *
     * @param test_ds
     *          Test dataset.
     * @param random_seed
     *
     * @return  Result/score.
     */
    double
    test(const dataset& test_ds, unsigned random_seed) {
      result_map_.clear();
      return add_results(test_ds, "test ");
    } // test

    // Protected data
    // =========================================================================
  protected:

    const crf_graph_type& structure;

    crf_model<F> model;

    crf_parameter_learner_parameters cpl_params;

    //! Indicates if the weights of model are good for initializing learning.
    bool use_weights;

    // Protected methods
    // =========================================================================

    void train_model(const dataset& ds, unsigned random_seed) {
      cpl_params.random_seed = random_seed;
      if (model.size() != 0) {
        crf_parameter_learner<F> cpl(model, ds, use_weights, cpl_params);
        model = cpl.current_model();
      } else {
        crf_parameter_learner<F> cpl(structure, ds, cpl_params);
        model = cpl.current_model();
      }
    }

    //! Compute results from model, and store them in result_map_.
    //! @param prefix  Prefix to add to result names.
    //! @return  Main result/score.
    double add_results(const dataset& ds, const std::string& prefix) {
      double ll = model.expected_log_likelihood(ds);
      result_map_[prefix + "log likelihood"] = ll;
      result_map_[prefix + "per-label accuracy"] =
        model.expected_per_label_accuracy(ds);
      return ll;
    }

  }; // class crf_validation_functor<F>

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CRF_VALIDATION_FUNCTOR_HPP
