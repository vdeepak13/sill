
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
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class multiclass_classifier : public singlelabel_classifier<LA> {

    // Public types
    //==========================================================================
  public:

    typedef singlelabel_classifier<LA> base;

    typedef typename base::la_type            la_type;
    typedef typename base::record_type        record_type;
    typedef typename base::value_type         value_type;
    typedef typename base::vector_type        vector_type;
    typedef typename base::matrix_type        matrix_type;
    typedef typename base::dense_vector_type  dense_vector_type;
    typedef typename base::dense_matrix_type  dense_matrix_type;

  private:
    static_assert(std::numeric_limits<double>::has_infinity);

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
  public:

    multiclass_classifier() : base() { }

    explicit multiclass_classifier(const datasource& ds)
      : base(ds) { }

    virtual ~multiclass_classifier() { }

    //! Train a new multiclass classifier of this type with the given data.
    virtual boost::shared_ptr<multiclass_classifier>
    create(dataset_statistics<la_type>& stats) const = 0;

    //! Train a new multiclass classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    virtual boost::shared_ptr<multiclass_classifier>
    create(oracle<la_type>& o, size_t n) const = 0;

    // Getters and helpers
    //==========================================================================

    //! Returns the number of classes (values the label can take).
    size_t nclasses() const;

    //! @return <log likelihood, std dev of log likelihood> estimated using
    //!         given test data
    std::pair<value_type, value_type>
    test_log_likelihood(const dataset<la_type>& testds,
                        value_type base = exp(1.)) const;

    //! @param n  max number of examples to be drawn from the given oracle
    //! @return <log likelihood, std dev of log likelihood> estimated using
    //!         given test data
    std::pair<value_type, value_type>
    test_log_likelihood(oracle<la_type>& o, size_t n,
                        value_type base = exp(1.)) const;

    // Prediction methods
    //==========================================================================

    using base::predict;

    //! Values indicating the confidences in each label, with
    //!  predict() == max_index(confidence()).
    //! If the classifier does not have actual confidence ratings,
    //!  then this returns values -1 or +1.
    virtual dense_vector_type confidences(const record_type& example) const;

    virtual dense_vector_type confidences(const assignment& example) const;

    //! Returns a prediction whose value indicates the label
    //! (like confidence()) but whose magnitude may differ.
    //! For boosters, this is the weighted sum of base predictions.
    //! If this is not implemented, then it returns confidence(example).
    virtual dense_vector_type
    predict_raws(const record_type& example) const {
      return confidences(example);
    }

    virtual dense_vector_type predict_raws(const assignment& example) const {
      return confidences(example);
    }

    //! Predict the probability of the class variable having each value.
    //! If this is not implemented, then it returns 1 for the label
    //! given by predict(example) and 0 for the other labels.
    virtual dense_vector_type
    probabilities(const record_type& example) const;

    virtual dense_vector_type probabilities(const assignment& example) const;

    // Methods for iterative learners
    // (None of these are implemented by non-iterative learners.)
    //==========================================================================

    using base::train_accuracies;
    using base::test_accuracies;

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    // Protected data members
    //==========================================================================
  protected:

    using base::label_;
    using base::label_index_;

  }; // class multiclass_classifier


  //============================================================================
  // Implementations of methods in multiclass_classifier
  //============================================================================


  // Getters and helpers
  //==========================================================================

  template <typename LA>
  size_t multiclass_classifier<LA>::nclasses() const {
    if (label_ == NULL) {
      assert(false);
      return 0;
    }
    return label_->size();
  }

  template <typename LA>
  std::pair<typename multiclass_classifier<LA>::value_type,
            typename multiclass_classifier<LA>::value_type>
  multiclass_classifier<LA>::test_log_likelihood(const dataset<la_type>& testds,
                                                 value_type base) const {
    if (testds.size() == 0) {
      std::cerr << "multiclass_classifier::test_log_likelihood() called with"
                << " no data." << std::endl;
      assert(false);
      return std::make_pair(0,0);
    }
    value_type loglike(0);
    value_type stddev(0);
    typename dataset<la_type>::record_iterator_type testds_end = testds.end();
    for (typename dataset<la_type>::record_iterator_type testds_it = testds.begin();
         testds_it != testds_end; ++testds_it) {
      const record_type& example = *testds_it;
      value_type ll(probabilities(example)[label(example)]);
      if (ll == 0) {
        loglike = - std::numeric_limits<value_type>::infinity();
        stddev = std::numeric_limits<value_type>::infinity();
        break;
      }
      ll = std::log(ll);
      loglike += ll;
      stddev += ll * ll;
    }
    loglike /= std::log(base);
    stddev /= (std::log(base) * std::log(base));
    if (testds.size() == 1)
      return std::make_pair(loglike,
                            std::numeric_limits<value_type>::infinity());
    stddev = sqrt((stddev - loglike * loglike / testds.size())
                  / (testds.size() - 1));
    loglike /= testds.size();
    return std::make_pair(loglike, stddev);
  }

  template <typename LA>
  std::pair<typename multiclass_classifier<LA>::value_type,
            typename multiclass_classifier<LA>::value_type>
  multiclass_classifier<LA>::test_log_likelihood(oracle<la_type>& o, size_t n,
                                                 value_type base) const {
    size_t cnt(0);
    value_type loglike(0);
    value_type stddev(0);
    while (cnt < n) {
      if (!(o.next()))
        break;
      const record_type& example = o.current();
      value_type ll(probabilities(example)[label(example)]);
      if (ll == 0) {
        loglike = - std::numeric_limits<value_type>::infinity();
        stddev = std::numeric_limits<value_type>::infinity();
        break;
      }
      ll = std::log(ll);
      loglike += ll;
      stddev += ll * ll;
      ++cnt;
    }
    loglike /= std::log(base);
    stddev /= (std::log(base) * std::log(base));
    if (cnt == 0) {
      std::cerr << "multiclass_classifier::test_log_likelihood() called with"
                << " an oracle with no data."
                << std::endl;
      assert(false);
      return std::make_pair(0,0);
    } else if (cnt == 1)
      return std::make_pair(loglike,
                            std::numeric_limits<value_type>::infinity());
    stddev = sqrt((stddev - loglike * loglike / cnt)/(cnt - 1));
    loglike /= cnt;
    return std::make_pair(loglike, stddev);
  }

  // Prediction methods
  //==========================================================================

  template <typename LA>
  typename multiclass_classifier<LA>::dense_vector_type
  multiclass_classifier<LA>::confidences(const record_type& example) const {
    dense_vector_type v(nclasses(), -1);
    size_t j(predict(example));
    v[j] = 1;
    return v;
  }

  template <typename LA>
  typename multiclass_classifier<LA>::dense_vector_type
  multiclass_classifier<LA>::confidences(const assignment& example) const {
    dense_vector_type v(nclasses(), -1);
    size_t j(predict(example));
    v[j] = 1;
    return v;
  }

  template <typename LA>
  typename multiclass_classifier<LA>::dense_vector_type
  multiclass_classifier<LA>::probabilities(const record_type& example) const {
    dense_vector_type v(nclasses(), 0);
    size_t j(predict(example));
    v[j] = 1;
    return v;
  }

  template <typename LA>
  typename multiclass_classifier<LA>::dense_vector_type
  multiclass_classifier<LA>::probabilities(const assignment& example) const{
    dense_vector_type v(nclasses(), 0);
    size_t j(predict(example));
    v[j] = 1;
    return v;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_CLASSIFIER_HPP
