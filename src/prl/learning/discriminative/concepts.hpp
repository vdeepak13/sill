
#ifndef SILL_LEARNING_DISCRIMINATIVE_CONCEPTS_HPP
#define SILL_LEARNING_DISCRIMINATIVE_CONCEPTS_HPP

#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/dataset/statistics.hpp>

#include <sill/macros_def.hpp>

/**
 * \file concepts.hpp Concepts for discriminative models
 *
 * Todo, as necessary: Add concepts for:
 *  - regression on real variables
 *  - multilabel regression
 */

namespace sill {

  /**
   * Optimization criteria (unnormalized scores, normalized scores, confidences)
   * These classes define methods for computing optimization criteria.
   * The criteria are meant to be MAXIMIZED.
   * These may be used as template parameters to any domain-partitioning
   * learners (learners which partition a domain and then
   * assign labels to partitions according to their majority classes).
   * (See, e.g., Schapire & Singer, 1999.)
   * Note: Right now, these are for binary labels only.
   */
  template <typename C>
  struct DomainPartitioningObjective {

    //! Indicates if the predictions are confidence-rated.
    static bool confidence_rated();

    /**
     * Compute normalized objective for binary partitioning.
     * This should be called separately for each partition.
     * @param rightA  weight of correctly classified examples in partition A
     * @param wrongA  weight of incorrectly classified examples in partition A
     * @param rightB  weight of correctly classified examples in partition B
     * @param wrongB  weight of incorrectly classified examples in partition B
     */
    static double objective(double rightA, double wrongA,
                            double rightB, double wrongB);

    //! Compute unnormalized objective for binary partitioning. Since this is
    //! unnormalized, it may be called separately for each partition.
    static double unnormalized(double right, double wrong);

    /**
     * Compute confidence of prediction for binary partitioning; the sign of
     * the confidence indicates the predicted class.
     * @param class0  weight of examples in partition with label 0
     * @param class1  weight of examples in partition with label 1
     */
    static double confidence(double class0, double class1);

    concept_usage(DomainPartitioningObjective) {
      sill::same_type(confidence_rated(), b);
      sill::same_type(objective(d,d,d,d), d);
      sill::same_type(unnormalized(d,d), d);
      sill::same_type(confidence(d,d), d);
    }

  private:

    bool b;
    double d;

  };

  ///////////////////////// BASE CLASSIFIERS /////////////////////////////

  /**
   * Concept class for multiclass classifiers.
   */
  template <typename C>
  struct MulticlassClassifier {

    //! Returns the finite variables in their natural order
    //! (including class variable).
    const finite_var_vector& finite_list() const;

    //! Returns the vector variables in their natural order
    const vector_var_vector& vector_list() const;

    //! Returns the class variable
    const finite_variable* label() const;

    //! Predict the label of a new example.
    size_t predict(const record& example) const;

    //! Predict the label of a new example.
    size_t predict(const assignment& example) const;

    //! Return training accuracy (or estimate of it)
    double train_accuracy() const;

    concept_usage(MulticlassClassifier) {
      sill::same_type(finite_list(), fvarvec);
      sill::same_type(vector_list(), vvarvec);
      sill::same_type(label(), l);
      sill::same_type(predict(r), i);
      sill::same_type(predict(example), i);
      sill::same_type(train_accuracy(), d);
    }

  private:

    const finite_var_vector& fvarvec;
    const vector_var_vector& vvarvec;
    const finite_variable* l;
    const record& r;
    const assignment& example;
    size_t i;
    double d;

  };

  /**
   * Concept class for binary classifiers.
   * Note this is a more specific concept than MulticlassClassifier.
   */
  template <typename C>
  struct BinaryClassifier : public MulticlassClassifier<C> {

    //! Indicates if the predictions are confidence-rated.
    static bool confidence_rated();

    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should return -1,+1.
    double confidence(const record& example) const;

    double confidence(const assignment& example) const;

    concept_usage(BinaryClassifier) {
      sill::same_type(confidence_rated(), b);
      sill::same_type(confidence(r), d);
      sill::same_type(confidence(example), d);
    }

  private:

    bool b;
    const record& r;
    const assignment& example;
    double d;

  };

  template <typename C>
  struct MultilabelClassifier {

    //! Returns the finite variables in their natural order
    //! (including class variable).
    const finite_var_vector& finite_list() const;

    //! Returns the vector variables in their natural order
    const vector_var_vector& vector_list() const;

    //! Returns the class variables
    sill::forward_range<const finite_variable*> labels() const;

    //! Predict the labels of a new example.
    sill::forward_range<size_t> predict(const record& example) const;

    //! Predict the labels of a new example.
    sill::forward_range<size_t> predict(const assignment& example) const;

    //! Predict the value of label variable var of a new example.
    size_t predict(const record& example, finite_variable* var) const;

    //! Predict the value of label variable var of a new example.
    size_t predict(const assignment& example, finite_variable* var) const;

    concept_usage(MultilabelClassifier) {
      sill::same_type(finite_list(), varvec);
      sill::same_type(vector_list(), varvec);
      sill::same_type(labels(), ls);
      sill::same_type(predict(r), vals);
      sill::same_type(predict(example), vals);
      sill::same_type(predict(r, var), val);
      sill::same_type(predict(example, var), val);
    }

  private:

    const var_vector& varvec;
    sill::forward_range<const finite_variable*> ls;
    const record& r;
    const assignment& example;
    sill::forward_range<size_t> vals;
    finite_variable* var;
    size_t val;

  };

  template <typename R>
  struct MulticlassRegressor {

    //! Returns the finite variables in their natural order
    const finite_var_vector& finite_list() const;

    //! Returns the vector variables in their natural order
    //! (including class variable).
    const vector_var_vector& vector_list() const;

    //! Returns the class variable
    const finite_variable* label() const;

    //! Predict the probability of the class variable having the given value.
    double probability(const record& example, size_t value) const;

    double probability(const assignment& example, size_t value) const;

    //! Predict the probabilities of the class variable having all
    //! potential values.
    sill::forward_range<double> probabilities(const record& example) const;

    sill::forward_range<double>
    probabilities(const assignment& example) const;

    concept_usage(MulticlassRegressor) {
      sill::same_type(finite_list(), fvarvec);
      sill::same_type(vector_list(), vvarvec);
      sill::same_type(label(), l);
      sill::same_type(probability(r, i), d);
      sill::same_type(probability(example, i), d);
      sill::same_type(probabilities(r), drange);
      sill::same_type(probabilities(example), drange);
    }

  private:

    const finite_var_vector& fvarvec;
    const vector_var_vector& vvarvec;
    const finite_variable* l;
    const record& r;
    const assignment& example;
    size_t i;
    double d;
    sill::forward_range<double> drange;

  };

  template <typename R>
  struct BinaryRegressor : MulticlassRegressor<R> {

    //! Predict the probability of the class variable having value 1.
    double probability(const record& example) const;

    double probability(const assignment& example) const;

    concept_usage(BinaryRegressor) {
      sill::same_type(probability(r), d);
      sill::same_type(probability(example), d);
    }

  private:

    const record& r;
    const assignment& example;
    double d;

  };

  //! Concept class for batch learners.
  template <typename C>
  struct BatchLearner {

    typedef typename C::parameters parameters;

    // TODO: Is there a way to put templated functions in concept_usage()?
    BatchLearner(const statistics& stats, parameters params);

  };

  //! Concept class for online learners.
  template <typename C>
  struct OnlineLearner {

    typedef typename C::parameters parameters;

    OnlineLearner(const oracle& oracle, parameters params);

  };

  ////////////////////// MAIN BASE CLASSIFIER CONCEPTS ///////////////////

  //! Concept class for batch binary classifiers.
  template <typename C>
  struct BatchBinaryClassifier
    : public BinaryClassifier<C>, public BatchLearner<C> { };

  //! Concept class for online binary classifiers.
  template <typename C>
  struct OnlineBinaryClassifier
    : public BinaryClassifier<C>, public OnlineLearner<C> { };

  //! Concept class for batch multiclass classifiers.
  template <typename C>
  struct BatchMulticlassClassifier
    : public MulticlassClassifier<C>, public BatchLearner<C> { };

  //! Concept class for online multiclass classifiers.
  template <typename C>
  struct OnlineMulticlassClassifier
    : public MulticlassClassifier<C>, public OnlineLearner<C> { };

  //! Concept class for batch multilabel classifiers.
  template <typename C>
  struct BatchMultilabelClassifier
    : public MultilabelClassifier<C>, public BatchLearner<C> { };

  //! Concept class for online multilabel classifiers.
  template <typename C>
  struct OnlineMultilabelClassifier
    : public MultilabelClassifier<C>, public OnlineLearner<C> { };

  ////////////////////// MAIN BASE REGRESSOR CONCEPTS ///////////////////

  //! Concept class for batch binary regressors.
  template <typename C>
  struct BatchBinaryRegressor
    : public BinaryRegressor<C>, public BatchLearner<C> { };

  //! Concept class for online binary regressors.
  template <typename C>
  struct OnlineBinaryRegressor
    : public BinaryRegressor<C>, public OnlineLearner<C> { };

  //! Concept class for batch multiclass regressors.
  template <typename C>
  struct BatchMulticlassRegressor
    : public MulticlassRegressor<C>, public BatchLearner<C> { };

  //! Concept class for online multiclass regressors.
  template <typename C>
  struct OnlineMulticlassRegressor
    : public MulticlassRegressor<C>, public OnlineLearner<C> { };

  ///////////////////////////// BOOSTER CONCEPTS //////////////////////////

  /**
   * Concept class for boosters (classifiers which make use of sub-classifiers).
   * Note this could be used for classifiers other than traditional boosting
   * algorithms.
   * @todo Should we derive specific types of booster concept classes or
   *       leave that to the user?
   */
  template <typename B>
  struct Booster {

    //! Type of weak learner (sub-classifier)
    typedef typename B::weak_learner weak_learner;

    //! Run next iteration of boosting.
    bool step();

    //! Returns the current iteration number (from 0)
    //!  (i.e., the number of boosting iterations completed).
    size_t iteration() const;

    concept_usage(Booster) {
      sill::same_type(step(), b);
      sill::same_type(iteration(), i);
    }

  private:

    bool b;
    size_t i;

  }; // struct Booster

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LEARNING_DISCRIMINATIVE_CONCEPTS_HPP
