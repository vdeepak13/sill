#ifndef SILL_MODEL_VALIDATION_FUNCTOR_HPP
#define SILL_MODEL_VALIDATION_FUNCTOR_HPP

#include <boost/timer.hpp>

#include <sill/learning/dataset/dataset.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Interface for model validation functors to be used with
   * validation_framework.
   *
   * A model validation functor has functions for training a model
   * and computing results.
   *
   * NOTE: This automatically records training and testing times.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class model_validation_functor {

    // Public types and structs
    // =========================================================================
  public:

    typedef LA la_type;

//    typedef record<LA> record_type;

    typedef typename la_type::vector_type vector_type;
    typedef typename la_type::value_type  value_type;
    typedef arma::Col<value_type>         dense_vector_type;

    /**
     * Enumeration over common scores.
     */
    enum model_score_enum { LOG_LIKELIHOOD,
                            PER_LABEL_ACCURACY,
                            ACCURACY,
                            TIME };

    // Constructors and destructors
    // =========================================================================

    model_validation_functor()
      : use_weights(false) { }

    virtual ~model_validation_functor() { }

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
     *          NOTE: If true, then the previously trained model was trained
     *                using the same dataset (when this method is called by
     *                param_list_validation_functor::test).
     * @param random_seed
     *
     * @return  Result/score.
     */
    virtual value_type
    train(const dataset<la_type>& train_ds, const dataset<la_type>& test_ds,
          const dense_vector_type& validation_params,
          bool warm_start_recommended,
          unsigned random_seed = time(NULL)) {
      use_weights = warm_start_recommended;
      result_map_.clear();

      timer.restart();
      train_model(train_ds, validation_params, random_seed);
      double train_time = timer.elapsed();
      add_results(train_ds, "train ");
      result_map_["train time"] = train_time;

      timer.restart();
      double res = add_results(test_ds, "test ");
      double test_time = timer.elapsed();
      result_map_["test time"] = test_time;
      return res;
    }

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
    virtual value_type
    train(const dataset<la_type>& train_ds, const dataset<la_type>& test_ds,
          unsigned random_seed = time(NULL)) {
      result_map_.clear();

      timer.restart();
      train_model(train_ds, random_seed);
      double train_time = timer.elapsed();
      add_results(train_ds, "train ");
      result_map_["train time"] = train_time;

      timer.restart();
      double res = add_results(test_ds, "test ");
      double test_time = timer.elapsed();
      result_map_["test time"] = test_time;
      return res;
    }

    /**
     * Compute results on the test dataset.
     *
     * @param test_ds
     *          Test dataset.
     * @param random_seed
     *
     * @return  Result/score.
     */
    virtual value_type
    test(const dataset<la_type>& test_ds, unsigned random_seed = time(NULL)) {
      result_map_.clear();
      timer.restart();
      double res = add_results(test_ds, "test ");
      double test_time = timer.elapsed();
      result_map_["test time"] = test_time;
      return res;
    }

    /**
     * Map: result type label --> result value
     */
    const std::map<std::string, value_type>& result_map() const {
      return result_map_;
    }

    // Protected data
    // =========================================================================
  protected:

    boost::timer timer;

    /**
     * Map: result type label --> result value
     * E.g., "test log likelihood" --> log likelihood value on test dataset
     */
    std::map<std::string, value_type> result_map_;

    //! Indicates if the weights of model are good for initializing learning.
    bool use_weights;

    // Protected methods
    // =========================================================================

    //! Train the model.
    //! This is expected to set some stored model (and handle warm starts).
    virtual void
    train_model(const dataset<la_type>& ds, unsigned random_seed) = 0;

    //! Train the model using the given parameters (e.g., regularization).
    //! This is expected to set some stored model (and handle warm starts).
    virtual void
    train_model(const dataset<la_type>& ds,
                const dense_vector_type& validation_params,
                unsigned random_seed) = 0;

    //! Compute results from model, and store them in result_map_.
    //! @param prefix  Prefix to add to result names.
    //! @return  Main result/score.
    virtual double
    add_results(const dataset<la_type>& ds, const std::string& prefix) = 0;

  }; // class model_validation_functor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_MODEL_VALIDATION_FUNCTOR_HPP
