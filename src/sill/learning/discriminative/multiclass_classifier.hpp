
#ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_CLASSIFIER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_CLASSIFIER_HPP

#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/discriminative/singlelabel_classifier.hpp>

#include <sill/macros_def.hpp>

/**
 * \file multiclass_classifier.hpp Multiclass Classifier Interface
 *
 * MOVE THIS TO SUBPAGE_DISCRIMINATIVE
 *
 * Discriminative Interface Groups:
 *  - By types of labels:
 *     - binary: 1 two-valued label
 *     - multiclass: 1 k-valued label
 *     - multilabel: multiple finite-valued labels
 *     - real: multiple real-valued labels
 *     - Note: It is tempting to think of having some of these inherit
 *             from each other, but it is not a good idea.
 *             Technically, none should inherit from the others.
 *             Plus, you get annoying diamond-shaped inheritance graphs.
 *  - By classification vs. regression:
 *     - *classifier: classification
 *        - All classifiers implement class conditional probability
 *          estimation, though ones which do not really support it
 *          simply return 0 or 1 based on the classification functions.
 *     - *regressor: general regression
 *  - For each type of learner:
 *     - (learner): classification/regression function
 *     - (learner)_engine: engine which holds data and trains a
 *       classifier/regressor
 *        - This is mainly for supporting online and iteratively trained
 *          learners (like boosters).
 *     - (learner)_shell: class which can produce an engine or
 *       classifier/regressor
 *  - Batch vs. online:
 *     - Batch and online learners are implemented in the same way and,
 *       in fact, may be implemented by the same class.  Online learners
 *       and iterative batch learners are treated the same way, where
 *       the learner engine holds a datasource, can be trained iteratively,
 *       and can be used as a classifier/regressor at any time.
 *  - Base learners vs. meta-learners
 *     - Meta-learners are given base learners via shared_ptrs in their
 *       parameters, and they are otherwise treated in the same way.
 *
 * Class hierarchy for classifiers:
 *  - multiclass_classifier
 *  - multiclass_classifier_engine : public multiclass_classifier
 *  - multiclass_classifier_shell
 *  - binary_classifier
 *  - binary_classifier_engine : public binary_classifier
 *  - binary_classifier_shell
 *  - multilabel_classifier
 *  - multilabel_classifier_engine : public multilabel_classifier
 *  - multilabel_classifier_shell
 *  - regressor
 *  - regressor_engine : public regressor
 *  - regressor_shell
 *  - multilabel_regressor
 *  - multilabel_regressor_engine : public multilabel_regressor
 *  - multilabel_regressor_shell
 *
 * Parameters:
 *  - Each learner has certain parameters, and a learner inherits
 *    parameters from its parents.  For example, stump_parameters inherits
 *    from binary_classifier_parameters.
 *
 * Designing classes (temp):
 *  - multiclass_classifier
 *     - random_seed
 *     - helper functions: label, label_index
 *     - is_confidence_rated
 *     - predict, confidence, predict_raw, probability
 *     - train_accuracy
 *     - save, load
 *     - name
 *  - multiclass_classifier_engine : public multiclass_classifier
 *     - engine_random_seed
 *     - current
 *     - step
 *     - set_confidences
 *     - is_iterative, is_online
 *     - reset_oracle, reset_stats
 *     - save_engine, load_engine
 *  - multiclass_classifier_shell
 *     - create_engine
 *     - create_classifier
 *     - parameters, reset_parameters (X -- difficult to inherit)
 *     - save_shell, load_shell
 *
 * Re-designing classes to get rid of engine & shell:
 *  - multiclass_classifier
 *  - binary_classifier
 *  - iterative_learner
 *  - iterative_multiclass_classifier
 *        : public multiclass_classifier, public iterative_learner
 *  - iterative_binary_classifier
 *        : public binary_classifier, public iterative_learner
 */

namespace sill {

  /**
   * Multiclass classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   */
  class multiclass_classifier : public singlelabel_classifier {

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

    // Constructors and destructors
    //==========================================================================

    multiclass_classifier() : base() { }

    explicit multiclass_classifier(const datasource& ds)
      : base(ds) { }

    virtual ~multiclass_classifier() { }

    //! Train a new multiclass classifier of this type with the given data.
    virtual boost::shared_ptr<multiclass_classifier>
    create(dataset_statistics& stats) const = 0;

    //! Train a new multiclass classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    virtual boost::shared_ptr<multiclass_classifier>
    create(oracle& o, size_t n) const = 0;

    // Getters and helpers
    //==========================================================================

    //! Returns the number of classes (values the label can take).
    size_t nclasses() const;

    //! @return <log likelihood, std dev of log likelihood> estimated using
    //!         given test data
    std::pair<double, double> test_log_likelihood(const dataset& testds,
                                                  double base = exp(1.)) const;

    //! @param n  max number of examples to be drawn from the given oracle
    //! @return <log likelihood, std dev of log likelihood> estimated using
    //!         given test data
    std::pair<double, double> test_log_likelihood(oracle& o, size_t n,
                                                  double base = exp(1.)) const;

    // Prediction methods
    //==========================================================================

    //! Values indicating the confidences in each label, with
    //!  predict() == max_index(confidence()).
    //! If the classifier does not have actual confidence ratings,
    //!  then this returns values -1 or +1.
    virtual vec confidences(const record& example) const;

    virtual vec confidences(const assignment& example) const;

    //! Returns a prediction whose value indicates the label
    //! (like confidence()) but whose magnitude may differ.
    //! For boosters, this is the weighted sum of base predictions.
    //! If this is not implemented, then it returns confidence(example).
    virtual vec predict_raws(const record& example) const {
      return confidences(example);
    }

    virtual vec predict_raws(const assignment& example) const {
      return confidences(example);
    }

    //! Predict the probability of the class variable having each value.
    //! If this is not implemented, then it returns 1 for the label
    //! given by predict(example) and 0 for the other labels.
    virtual vec probabilities(const record& example) const;

    virtual vec probabilities(const assignment& example) const;

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

  }; // class multiclass_classifier

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_CLASSIFIER_HPP
