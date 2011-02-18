#ifndef SILL_MODEL_TRAINING_FUNCTOR_HPP
#define SILL_MODEL_TRAINING_FUNCTOR_HPP

#include <sill/learning/dataset/dataset.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Interface for model training functors to be used with validation_framework.
   *
   * A model training functor has a function train() which trains a model
   * and computes results.
   */
  class model_training_functor {

    // Constructors and destructors
    // =========================================================================
  public:

    model_training_functor() {
    }

    virtual ~model_training_functor() { }

    // Public methods
    // =========================================================================

    /**
     * Train a model, and compute results on the test set.
     * This method can set the result map if needed,
     * but only the returned value is required by validation_framework.
     *
     * @param train_ds                Training dataset.
     * @param test_ds                 Test dataset.
     * @param validation_params 
     *         Used for choosing parameters via the validation_framework.
     *         This is not given if the validation_framework is only being used
     *         to compute results, not choose parameters.
     * @param warm_start_recommended
     *         If true, then the validation_framework recommends that training
     *         be initialized using the most recently trained model.
     * @param random_seed
     *
     * @return  Result/score.
     */
    virtual double
    train(const dataset& train_ds, const dataset& test_ds,
          const vec& validation_params, bool warm_start_recommended,
          unsigned random_seed = time(NULL)) = 0;

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
     * E.g., "test log likelihood" --> log likelihood value on test set
     */
    std::map<std::string, double> result_map_;

  }; // class model_training_functor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_MODEL_TRAINING_FUNCTOR_HPP
