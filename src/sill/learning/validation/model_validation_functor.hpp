#ifndef SILL_MODEL_VALIDATION_FUNCTOR_HPP
#define SILL_MODEL_VALIDATION_FUNCTOR_HPP

#include <sill/learning/dataset/dataset.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Interface for model validation functors to be used with
   * validation_framework.
   *
   * A model validation functor has functions for training a model
   * and computing results.
   */
  class model_validation_functor {

    // Public types and structs
    // =========================================================================
  public:

    /**
     * Enumeration over common scores.
     */
    enum model_score_enum { LOG_LIKELIHOOD,
                            PER_LABEL_ACCURACY,
                            ACCURACY,
                            TIME };

    // Constructors and destructors
    // =========================================================================

    model_validation_functor() { }

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
     * @param random_seed
     *
     * @return  Result/score.
     */
    virtual double
    train(const dataset& train_ds, const dataset& test_ds,
          const vec& validation_params, bool warm_start_recommended,
          unsigned random_seed) = 0;

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
    virtual double
    train(const dataset& train_ds, const dataset& test_ds,
          unsigned random_seed) = 0;

    /**
     * Compute results on the test dataset.
     *
     * @param test_ds
     *          Test dataset.
     * @param random_seed
     *
     * @return  Result/score.
     */
    virtual double
    test(const dataset& test_ds, unsigned random_seed) = 0;

    /**
     * Map: result type label --> result value
     */
    const std::map<std::string, double>& result_map() const {
      return result_map_;
    }

    // Protected data
    // =========================================================================
  protected:

    /**
     * Map: result type label --> result value
     * E.g., "test log likelihood" --> log likelihood value on test dataset
     */
    std::map<std::string, double> result_map_;

  }; // class model_validation_functor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_MODEL_VALIDATION_FUNCTOR_HPP
