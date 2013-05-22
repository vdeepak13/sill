
#ifndef SILL_LEARNING_DISCRIMINATIVE_BINARY_CLASSIFIER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BINARY_CLASSIFIER_HPP

#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/discriminative/singlelabel_classifier.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Binary classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class binary_classifier : public singlelabel_classifier<LA> {

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
    create(dataset_statistics<la_type>& stats) const = 0;

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    virtual boost::shared_ptr<binary_classifier>
    create(oracle<la_type>& o, size_t n) const = 0;

    // Prediction methods
    //==========================================================================

    using base::predict;

    //! Value indicating the confidence in label +1, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this returns values -1 or +1.
    virtual value_type confidence(const record_type& example) const {
      return (predict(example) == 0 ? -1 : 1);
    }

    virtual value_type confidence(const assignment& example) const {
      return (predict(example) == 0 ? -1 : 1);
    }

    //! Returns a prediction whose value indicates the label
    //! (like confidence()) but whose magnitude may differ.
    //! For boosters, this is the weighted sum of base predictions.
    //! If this is not implemented, then it returns confidence(example).
    virtual value_type predict_raw(const record_type& example) const {
      return confidence(example);
    }

    virtual value_type predict_raw(const assignment& example) const {
      return confidence(example);
    }

    //! Predict the probability of the class variable having value +1.
    //! If this is not implemented, then it returns predict(example).
    virtual value_type probability(const record_type& example) const {
      return predict(example);
    }

    virtual value_type probability(const assignment& example) const {
      return predict(example);
    }

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

  }; // class binary_classifier

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BINARY_CLASSIFIER_HPP
