
#ifndef PRL_LEARNING_DISCRIMINATIVE_CLASSIFIER_HPP
#define PRL_LEARNING_DISCRIMINATIVE_CLASSIFIER_HPP

#include <prl/learning/discriminative/learner.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   */
  class classifier : public learner {

    typedef learner base;

    // Public methods
    //==========================================================================
  public:

    // Virtual methods from base classes (*means pure virtual):
    //  From learner:
    //   name()*
    //   fullname()*
    //   is_online()*
    //   training_time()
    //   random_seed()
    //   save(), load()

    // Constructors and destructors
    //==========================================================================

    virtual ~classifier() { }

    // Getters and helpers
    //==========================================================================

    //! Indicates if the predictions are confidence-rated.
    //! Note that confidence rating may be optimized for different objectives.
    //! This is false by default.
    virtual bool is_confidence_rated() const { return false; }

    // Learning and mutating operations
    //==========================================================================

    /**
     * For confidence-rated hypotheses, this sets the confidence-rated
     * outputs according to the sufficient statistics for a test set.
     * For non-confidenced-rated hypotheses, this still may change the
     * sign of the predictions.
     * This asserts false when it is not implemented.
     *
     * @param  ds  dataset used to estimate statistics needed
     * @return estimated error rate for given data and chosen confidences
     */
    virtual double set_confidences(const dataset& ds) {
      std::cerr << "classifier::set_confidences() has not been"
                << " implemented for this classifier!" << std::endl;
      assert(false);
      return - std::numeric_limits<double>::max();
    }

    /**
     * For confidence-rated hypotheses, this sets the confidence-rated
     * outputs according to the sufficient statistics for a test set.
     * For non-confidenced-rated hypotheses, this still may change the
     * sign of the predictions.
     * This asserts false when it is not implemented.
     *
     * @param o  data oracle used to estimate statistics needed
     * @param n  max number of examples to be drawn from the given oracle
     * @return estimated error rate for given data and chosen confidences
     */
    virtual double set_confidences(oracle& o, size_t n) {
      std::cerr << "classifier::set_confidences() has not been"
                << " implemented for this classifier!" << std::endl;
      assert(false);
      return - std::numeric_limits<double>::max();
    }

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

  }; // class classifier

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_CLASSIFIER_HPP
