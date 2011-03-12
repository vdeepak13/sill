
#ifndef SILL_LEARNING_DISCRIMINATIVE_MULTILABEL_CLASSIFIER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_MULTILABEL_CLASSIFIER_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/discriminative/classifier.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Multilabel classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class multilabel_classifier : public classifier<LA> {

    // Public types
    //==========================================================================
  public:

    typedef classifier<LA> base;

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
    create(dataset_statistics<la_type>& stats) const = 0;

    //! Train a new multilabel classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    virtual boost::shared_ptr<multilabel_classifier>
    create(oracle<la_type>& o, size_t n) const = 0;

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
    void
    label(const dataset<la_type>& ds, size_t i, std::vector<size_t>& v) const;

    void
    label(const dataset<la_type>& ds, size_t i, dense_vector_type& v) const;

    //! Returns the values of the class variables in the given record,
    //! via the vector v.
    //! Warning: This does not check if the label_index is valid.
    //! @param v  Vector in which to store the label values.
    void label(const record_type& r, std::vector<size_t>& v) const;

    void label(const record_type& r, dense_vector_type& v) const;

    //! Returns the values of the class variables in the given assignment,
    //! via the vector v.
    //! Warning: This does not check if the label is valid.
    //! @param v  Vector in which to store the label values.
    void label(const assignment& example, std::vector<size_t>& v) const;

    void label(const assignment& example, dense_vector_type& v) const;

    //! Sets the given assignment fa to have the label values for the finite
    //! values given.
    //! This does not affect non-label values in the assignment.
    void
    assign_labels(const std::vector<size_t>& r, finite_assignment& fa) const;

    //! Returns the classification accuracy for each label.
    dense_vector_type test_accuracy(const dataset<la_type>& testds) const;

    //! This uses the joint probabilities, not the per-label marginals.
    //! @return <log likelihood, std dev of log likelihood> estimated using
    //!         given test data
    std::pair<value_type, value_type>
    test_log_likelihood(const dataset<la_type>& testds,
                        value_type base = exp(1.)) const {
      if (testds.size() == 0) {
        std::cerr << "multilabel_classifier::test_log_likelihood() called with"
                  << " an empty dataset." << std::endl;
        assert(false);
        return std::make_pair(0,0);
      }
      double loglike(0);
      double stddev(0);
      typename dataset<la_type>::record_iterator testds_end = testds.end();
      finite_assignment fa;
      for (typename dataset<la_type>::record_iterator testds_it(testds.begin());
           testds_it != testds_end; ++testds_it) {
        const record_type& example = *testds_it;
        assign_labels(example.finite(), fa);
        double ll(probabilities(example).v(fa));
        if (ll == 0) {
          loglike = - std::numeric_limits<double>::infinity();
          stddev = std::numeric_limits<double>::infinity();
          break;
        }
        ll = std::log(ll);
        loglike += ll;
        stddev += ll * ll;
      }
      loglike /= std::log(base);
      stddev /= (std::log(base) * std::log(base));
      if (testds.size() == 1)
        return std::make_pair(loglike, std::numeric_limits<double>::infinity());
      stddev = sqrt((stddev - loglike * loglike / testds.size())
                    / (testds.size() - 1));
      loglike /= testds.size();
      return std::make_pair(loglike, stddev);
    }

    // Prediction methods
    //==========================================================================

    //! Predict the labels of a new example.
    //! @return  vector of label values
    std::vector<size_t> predict(const record_type& example) const;

    std::vector<size_t> predict(const assignment& example) const;

    //! Predict the labels of a new example, storing the predictions in
    //! the given vector/assignment v.
    virtual void
    predict(const record_type& example, std::vector<size_t>& v) const = 0;

    virtual void
    predict(const assignment& example, std::vector<size_t>& v) const = 0;

    virtual void
    predict(const record_type& example, finite_assignment& v) const = 0;

    virtual void
    predict(const assignment& example, finite_assignment& v) const = 0;

    //! Values indicating the confidences in each class, with
    //!  predict()[j] == max_index(confidence()[j]).
    //! If the classifier does not have actual confidence ratings,
    //!  then this returns values -1 or +1.
    virtual std::vector<dense_vector_type>
    confidences(const record_type& example) const;

    virtual std::vector<dense_vector_type>
    confidences(const assignment& example) const;

    //! For each label, returns a prediction whose value indicates the class
    //! (like confidence()) but whose magnitude may differ.
    //! For boosters, this is the weighted sum of base predictions.
    //! If this is not implemented, then it returns confidences(example).
    virtual std::vector<dense_vector_type>
    predict_raws(const record_type& example) const {
      return confidences(example);
    }

    virtual std::vector<dense_vector_type>
    predict_raws(const assignment& example) const {
      return confidences(example);
    }

    //! For each label, predict the marginal probability of each class.
    //! If this is not implemented, then it returns 1 for the label
    //! given by predict(example) and 0 for the other labels.
    virtual std::vector<dense_vector_type>
    marginal_probabilities(const record_type& example) const;

    virtual std::vector<dense_vector_type>
    marginal_probabilities(const assignment& example) const;

    //! Returns a factor which gives the probability for each assignment
    //! to the labels.
    //! If this is not implemented, then it returns 1 for the assignment
    //! given by predict(example) and 0 for the other assignments.
    virtual table_factor probabilities(const record_type& example) const;

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

    // Protected data members
    //==========================================================================
  protected:

    //! Class variables
    finite_var_vector labels_;

    //! Indices of class variables in dataset's records
    std::vector<size_t> label_indices_;

  }; // class multilabel_classifier


  //============================================================================
  // Implementations of methods in multilabel_classifier
  //============================================================================


  // Getters and helpers
  //==========================================================================

  template <typename LA>
  std::vector<size_t> multilabel_classifier<LA>::nclasses() const {
    std::vector<size_t> v(labels_.size());
    for (size_t j = 0; j < labels_.size(); ++j)
      v[j] = labels_[j]->size();
    return v;
  }

  template <typename LA>
  void multilabel_classifier<LA>::
  label(const dataset<la_type>& ds, size_t i, std::vector<size_t>& v) const {
    v.resize(labels_.size());
    for (size_t j = 0; j < labels_.size(); ++j)
      v[j] = ds.finite(i,label_indices_[j]);
  }

  template <typename LA>
  void multilabel_classifier<LA>::
  label(const dataset<la_type>& ds, size_t i, dense_vector_type& v) const {
    v.resize(labels_.size());
    for (size_t j = 0; j < labels_.size(); ++j)
      v[j] = ds.finite(i,label_indices_[j]);
  }

  template <typename LA>
  void multilabel_classifier<LA>::
  label(const record_type& r, std::vector<size_t>& v) const {
    v.resize(labels_.size());
    for (size_t j = 0; j < labels_.size(); ++j)
      v[j] = r.finite(label_indices_[j]);
  }

  template <typename LA>
  void multilabel_classifier<LA>::label(const record_type& r, dense_vector_type& v) const {
    v.resize(labels_.size());
    for (size_t j = 0; j < labels_.size(); ++j)
      v[j] = r.finite(label_indices_[j]);
  }

  template <typename LA>
  void multilabel_classifier<LA>::
  label(const assignment& example, std::vector<size_t>& v) const {
    v.resize(labels_.size());
    const finite_assignment& fa = example.finite();
    for (size_t j = 0; j < labels_.size(); ++j)
      v[j] = safe_get(fa, labels_[j]);
  }

  template <typename LA>
  void multilabel_classifier<LA>::
  label(const assignment& example, dense_vector_type& v) const {
    v.resize(labels_.size());
    const finite_assignment& fa = example.finite();
    for (size_t j = 0; j < labels_.size(); ++j)
      v[j] = safe_get(fa, labels_[j]);
  }

  template <typename LA>
  void
  multilabel_classifier<LA>::
  assign_labels(const std::vector<size_t>& r, finite_assignment& fa) const {
    for (size_t j(0); j < labels_.size(); ++j)
      fa[labels_[j]] = r[label_indices_[j]];
  }

  template <typename LA>
  typename multilabel_classifier<LA>::dense_vector_type
  multilabel_classifier<LA>::
  test_accuracy(const dataset<la_type>& testds) const {
    dense_vector_type test_acc(nlabels(), 0);
    if (testds.size() == 0) {
      std::cerr << "multilabel_classifier::test_accuracy() called with an"
                << " empty dataset." << std::endl;
      assert(false);
      return test_acc;
    }
    typename dataset<la_type>::record_iterator testds_end = testds.end();
    std::vector<size_t> truth;
    for (typename dataset<la_type>::record_iterator testds_it = testds.begin();
         testds_it != testds_end; ++testds_it) {
      const record_type& example = *testds_it;
      std::vector<size_t> pred(predict(example));
      label(example, truth);
      for (size_t j(0); j < labels_.size(); ++j)
        if (pred[j] == truth[j])
          ++test_acc[j];
    }
    for (size_t j(0); j < labels_.size(); ++j)
      test_acc[j] /= testds.size();
    return test_acc;
  }

  // Prediction methods
  //==========================================================================

  template <typename LA>
  std::vector<size_t>
  multilabel_classifier<LA>::predict(const record_type& example) const {
    std::vector<size_t> preds(labels_.size());
    predict(example, preds);
    return preds;
  }

  template <typename LA>
  std::vector<size_t>
  multilabel_classifier<LA>::predict(const assignment& example) const {
    std::vector<size_t> preds(labels_.size());
    predict(example, preds);
    return preds;
  }

  template <typename LA>
  std::vector<typename multilabel_classifier<LA>::dense_vector_type>
  multilabel_classifier<LA>::confidences(const record_type& example) const {
    std::vector<dense_vector_type> c(labels_.size());
    std::vector<size_t> preds(predict(example));
    for (size_t j = 0; j < labels_.size(); ++j) {
      c[j].resize(labels_[j]->size(), -1);
      c[j][preds[j]] = 1;
    }
    return c;
  }

  template <typename LA>
  std::vector<typename multilabel_classifier<LA>::dense_vector_type>
  multilabel_classifier<LA>::confidences(const assignment& example) const {
    std::vector<dense_vector_type> c(labels_.size());
    std::vector<size_t> preds(predict(example));
    for (size_t j = 0; j < labels_.size(); ++j) {
      c[j].resize(labels_[j]->size(), -1);
      c[j][preds[j]] = 1;
    }
    return c;
  }

  template <typename LA>
  std::vector<typename multilabel_classifier<LA>::dense_vector_type>
  multilabel_classifier<LA>::
  marginal_probabilities(const record_type& example) const {
    std::vector<dense_vector_type> c(labels_.size());
    std::vector<size_t> preds(predict(example));
    for (size_t j = 0; j < labels_.size(); ++j) {
      c[j].resize(labels_[j]->size(), 0);
      c[j][preds[j]] = 1;
    }
    return c;
  }

  template <typename LA>
  std::vector<typename multilabel_classifier<LA>::dense_vector_type>
  multilabel_classifier<LA>::
  marginal_probabilities(const assignment& example) const {
    std::vector<dense_vector_type> c(labels_.size());
    std::vector<size_t> preds(predict(example));
    for (size_t j = 0; j < labels_.size(); ++j) {
      c[j].resize(labels_[j]->size(), 0);
      c[j][preds[j]] = 1;
    }
    return c;
  }

  template <typename LA>
  table_factor
  multilabel_classifier<LA>::probabilities(const record_type& example) const {
    table_factor f(labels_, 0);
    finite_assignment fa;
    predict(example, fa);
    f.set_v(fa, 1);
    return f;
  }

  template <typename LA>
  table_factor
  multilabel_classifier<LA>::probabilities(const assignment& example) const {
    table_factor f(labels_, 0);
    finite_assignment fa;
    predict(example, fa);
    f.set_v(fa, 1);
    return f;
  }

  // Save and load methods
  //==========================================================================

  template <typename LA>
  void multilabel_classifier<LA>::save(std::ofstream& out, size_t save_part,
                                   bool save_name) const {
    base::save(out, save_part, save_name);
    out << label_indices_ << "\n";
  }

  template <typename LA>
  bool
  multilabel_classifier<LA>::
  load(std::ifstream& in, const datasource& ds, size_t load_part) {
    std::string line;
    getline(in, line);
    std::istringstream is(line);
    read_vec(is, label_indices_);
    labels_.resize(label_indices_.size(), NULL);
    for (size_t j = 0; j < label_indices_.size(); ++j) {
      if (label_indices_[j] < ds.num_finite())
        labels_[j] = ds.finite_list()[label_indices_[j]];
      else {
        labels_[j] = NULL;
        assert(false);
        return false;
      }
    }
    return true;
  }

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTILABEL_CLASSIFIER_HPP
