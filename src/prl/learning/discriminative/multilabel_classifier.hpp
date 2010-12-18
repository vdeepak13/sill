
#ifndef SILL_LEARNING_DISCRIMINATIVE_MULTILABEL_CLASSIFIER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_MULTILABEL_CLASSIFIER_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/statistics.hpp>
#include <sill/learning/discriminative/classifier.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Multilabel classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   */
  class multilabel_classifier : public classifier {

    // Protected data members
    //==========================================================================
  protected:

    typedef classifier base;

    //! Class variables
    finite_var_vector labels_;

    //! Indices of class variables in dataset's records
    std::vector<size_t> label_indices_;

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
    //   test_accuracy()*
    //   set_confidences()

    // Constructors and destructors
    //==========================================================================

    multilabel_classifier() : base() { }

    explicit multilabel_classifier(const datasource& ds)
      : labels_(ds.finite_class_variables()) {
      assert(ds.vector_class_variables().size() == 0);
      assert(labels_.size() > 0);
      for (size_t j = 0; j < labels_.size(); ++j)
        label_indices_.push_back(ds.record_index(labels_[j]));
    }

    virtual ~multilabel_classifier() { }

    //! Train a new multilabel classifier of this type with the given data.
    virtual boost::shared_ptr<multilabel_classifier>
    create(statistics& stats) const = 0;

    //! Train a new multilabel classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    virtual boost::shared_ptr<multilabel_classifier>
    create(oracle& o, size_t n) const = 0;

    // Getters and helpers
    //==========================================================================

    //! Returns the number of labels.
    size_t nlabels() const {
      return labels_.size();
    }

    //! Returns the number of classes each label can take.
    std::vector<size_t> nclasses() const;

    //! Returns the class variables
    const finite_var_vector& labels() const {
      return labels_;
    }

    //! Returns the indices of the class variables in records' finite data
    const std::vector<size_t>& label_indices() const {
      return label_indices_;
    }

    //! Returns the values of the class variables for record i in the given
    //! dataset, via the vector v.
    //! Warning: This does not check if the label_index is valid.
    //! @param v  Vector in which to store the label values.
    void label(const dataset& ds, size_t i, std::vector<size_t>& v) const;

    void label(const dataset& ds, size_t i, vec& v) const;

    //! Returns the values of the class variables in the given record,
    //! via the vector v.
    //! Warning: This does not check if the label_index is valid.
    //! @param v  Vector in which to store the label values.
    void label(const record& r, std::vector<size_t>& v) const;

    void label(const record& r, vec& v) const;

    //! Returns the values of the class variables in the given assignment,
    //! via the vector v.
    //! Warning: This does not check if the label is valid.
    //! @param v  Vector in which to store the label values.
    void label(const assignment& example, std::vector<size_t>& v) const;

    void label(const assignment& example, vec& v) const;

    //! Sets the given assignment fa to have the label values for the finite
    //! values given.
    //! This does not affect non-label values in the assignment.
    void
    assign_labels(const std::vector<size_t>& r, finite_assignment& fa) const;

    //! Returns the classification accuracy for each label.
    vec test_accuracy(const dataset& testds) const;

    //! This uses the joint probabilities, not the per-label marginals.
    //! @return <log likelihood, std dev of log likelihood> estimated using
    //!         given test data
    std::pair<double, double> test_log_likelihood(const dataset& testds,
                                                  double base = exp(1.)) const;

    // Prediction methods
    //==========================================================================

    //! Predict the labels of a new example.
    //! @return  vector of label values
    std::vector<size_t> predict(const record& example) const;

    std::vector<size_t> predict(const assignment& example) const;

    //! Predict the labels of a new example, storing the predictions in
    //! the given vector/assignment v.
    virtual void
    predict(const record& example, std::vector<size_t>& v) const = 0;

    virtual void
    predict(const assignment& example, std::vector<size_t>& v) const = 0;

    virtual void
    predict(const record& example, finite_assignment& v) const = 0;

    virtual void
    predict(const assignment& example, finite_assignment& v) const = 0;

    //! Values indicating the confidences in each class, with
    //!  predict()[j] == max_index(confidence()[j]).
    //! If the classifier does not have actual confidence ratings,
    //!  then this returns values -1 or +1.
    virtual std::vector<vec> confidences(const record& example) const;

    virtual std::vector<vec> confidences(const assignment& example) const;

    //! For each label, returns a prediction whose value indicates the class
    //! (like confidence()) but whose magnitude may differ.
    //! For boosters, this is the weighted sum of base predictions.
    //! If this is not implemented, then it returns confidences(example).
    virtual std::vector<vec> predict_raws(const record& example) const {
      return confidences(example);
    }

    virtual std::vector<vec> predict_raws(const assignment& example) const {
      return confidences(example);
    }

    //! For each label, predict the marginal probability of each class.
    //! If this is not implemented, then it returns 1 for the label
    //! given by predict(example) and 0 for the other labels.
    virtual std::vector<vec>
    marginal_probabilities(const record& example) const;

    virtual std::vector<vec>
    marginal_probabilities(const assignment& example) const;

    //! Returns a factor which gives the probability for each assignment
    //! to the labels.
    //! If this is not implemented, then it returns 1 for the assignment
    //! given by predict(example) and 0 for the other assignments.
    virtual table_factor probabilities(const record& example) const;

    virtual table_factor probabilities(const assignment& example) const;

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    virtual void save(std::ofstream& out, size_t save_part = 0,
                      bool save_name = true) const;

    /**
     * Input the learner from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    virtual bool
    load(std::ifstream& in, const datasource& ds, size_t load_part);

  }; // class multilabel_classifier

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTILABEL_CLASSIFIER_HPP
