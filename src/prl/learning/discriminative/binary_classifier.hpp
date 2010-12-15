
#ifndef PRL_LEARNING_DISCRIMINATIVE_BINARY_CLASSIFIER_HPP
#define PRL_LEARNING_DISCRIMINATIVE_BINARY_CLASSIFIER_HPP

#include <prl/learning/dataset/statistics.hpp>
#include <prl/learning/discriminative/singlelabel_classifier.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Binary classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   */
  class binary_classifier : public singlelabel_classifier {

    // Protected data members
    //==========================================================================

    typedef singlelabel_classifier base;

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

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
    //  From classifier:
    //   is_confidence_rated()
    //   train_accuracy()
    //   set_confidences()
    //  From singlelabel_classifier:
    //   predict()*

    // Constructors
    //==========================================================================

    binary_classifier() : base() { }

    explicit binary_classifier(const datasource& ds)
      : base(ds) {
      assert(label_->size() == 2);
    }

    virtual ~binary_classifier() { }

    //! Train a new binary classifier of this type with the given data.
    virtual boost::shared_ptr<binary_classifier>
    create(statistics& stats) const = 0;

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    virtual boost::shared_ptr<binary_classifier>
    create(oracle& o, size_t n) const = 0;

    // Prediction methods
    //==========================================================================

    //! Value indicating the confidence in label +1, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this returns values -1 or +1.
    virtual double confidence(const record& example) const {
      return (predict(example) == 0 ? -1 : 1);
    }

    virtual double confidence(const assignment& example) const {
      return (predict(example) == 0 ? -1 : 1);
    }

    //! Returns a prediction whose value indicates the label
    //! (like confidence()) but whose magnitude may differ.
    //! For boosters, this is the weighted sum of base predictions.
    //! If this is not implemented, then it returns confidence(example).
    virtual double predict_raw(const record& example) const {
      return confidence(example);
    }

    virtual double predict_raw(const assignment& example) const {
      return confidence(example);
    }

    //! Predict the probability of the class variable having value +1.
    //! If this is not implemented, then it returns predict(example).
    virtual double probability(const record& example) const {
      return predict(example);
    }

    virtual double probability(const assignment& example) const {
      return predict(example);
    }

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

  }; // class binary_classifier

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_BINARY_CLASSIFIER_HPP
