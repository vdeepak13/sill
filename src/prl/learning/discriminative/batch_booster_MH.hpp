
#ifndef PRL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_MH_HPP
#define PRL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_MH_HPP

#include <prl/assignment.hpp>
#include <prl/datastructure/concepts.hpp>
#include <prl/learning/discriminative/concepts.hpp>

#include <prl/macros_def.hpp>

// Set to 1 to print out debugging information
#define BATCH_BOOSTER_MH_DEBUG 0

namespace prl {

  /**
   * Batch boosting algorithm for multiclass labels which uses binary weak
   * learners with confidence-rated predictions.
   *
   * TODO: fix the following comment:
   * To create, for example, batch AdaBoost with traditional decision trees,
   * construct:
   * Batch_Booster_MH<boosting::AdaBoost,
   *                 decision_tree<discriminative::criterion_exponential_loss> >
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @param BoosterCriterion optimization criterion defining the booster
   * @param WeakLearner      weak learning algorithm to boost
   * @todo serialization
   *
   * THIS IS INCOMPLETE
   */
  template <typename BoosterCriterion, typename WeakLearner>
  class batch_booster_mh {

    // Public type declarations
    //==========================================================================
  public:

    //! Type of weak learner being used.
    typedef WeakLearner weak_learner;

    //! Type of data source
    typedef dataset data_type;

    //! Type of data source used by weak learner
    typedef typename WeakLearner::data_type wl_data_type;

    //////////////// PARAMETERS /////////////////////////////////////////
    /**
     * Parameter names
     * - CONFIDENCE_SMOOTHING: double; value used for smoothing confidences
     *    (default = (2 * number_of_labels * number_of_datapoints)^-1)
     * - RANDOM_SEED: double; used to make the algorithm deterministic
     *    (default = random)
     * - RESAMPLING: std::size_t; if > 0, train WL with this many examples
     *    (default = 0)
     * - SCALE_RESAMPLING: bool; if true, scale value for RESAMPLING by log(t)
     *    where t is the number of rounds
     *    (default = false)
     */
    enum parameters {CONFIDENCE_SMOOTHING, RANDOM_SEED, RESAMPLING,
                     SCALE_RESAMPLING, PARAMETER_ARITY};

    // Public methods
    //==========================================================================

    // Constructors
    //==========================================================================

    batch_booster_mh(const statistics<vec>& stats,
                     const variable_h label, parameters params) {
    }

    // Getters and helpers
    //==========================================================================

    //! Returns the current iteration number (from 0)
    //!  (i.e., the number of boosting iterations completed).
    std::size_t iteration() const {
    }

    // Learning and mutating operations
    //==========================================================================

    //! Run next iteration of boosting.
    bool step() {
    }

    // Prediction methods
    //==========================================================================

    //! Predict the label of a new example.
    std::size_t predict(const assignment& example) const {
    }

  }; // class batch_booster_mh

} // namespace prl

#undef BATCH_BOOSTER_MH_DEBUG

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_MH_HPP
