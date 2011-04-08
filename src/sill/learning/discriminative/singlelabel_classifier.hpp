
#ifndef SILL_LEARNING_DISCRIMINATIVE_SINGLELABEL_CLASSIFIER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_SINGLELABEL_CLASSIFIER_HPP

#include <sill/learning/discriminative/classifier.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Single-label classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo I should create a general thing for doing confidence-rated
   *       predictions.  It should be easy to write a function to set
   *       confidences, etc.  It should probably be in this interface but
   *       implemented in binary_classifier and multiclass_classifier.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class singlelabel_classifier : public classifier<LA> {

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

    singlelabel_classifier()
      : base(), label_(NULL), label_index_(0) { }

    explicit singlelabel_classifier(const datasource& ds) {
      assert(ds.vector_class_variables().size() == 0);
      assert(ds.finite_class_variables().size() == 1);
      label_ = ds.finite_class_variables().front();
      label_index_ = ds.record_index(label_);
    }

    virtual ~singlelabel_classifier() { }

    // Getters and helpers
    //==========================================================================

    //! Returns the class variable
    finite_variable* label() const { return label_; }

    //! Returns the index of the class variable in records' finite data
    const size_t label_index() const { return label_index_; }

    //! Returns the value of the class variable for record i in the given
    //! dataset.
    //! Warning: This does not check if the label_index is valid.
    size_t label(const dataset<la_type>& ds, size_t i) const {
      return ds.finite(i,label_index_);
    }

    //! Returns the value of the class variable in the given record.
    //! Warning: This does not check if the label_index is valid.
    size_t label(const record_type& r) const {
      return r.finite(label_index_);
    }

    //! Returns the value of the class variable in the given assignment.
    //! Warning: This does not check if the label is valid.
    size_t label(const assignment& example) const {
      return safe_get(example.finite(), label_);
    }

    //! Returns training accuracy (or estimate of it).
    //! This is not guaranteed to be set (and = -1 if not set).
    virtual value_type train_accuracy() const { return -1; }

    //! Returns the accuracy on the given test data.
    value_type test_accuracy(const dataset<la_type>& testds) const {
      if (testds.size() == 0) {
        std::cerr << "singlelabel_classifier::test_accuracy() called with an"
                  << " empty dataset." << std::endl;
        assert(false);
        return -1;
      }
      value_type test_acc(0);
      typename dataset<la_type>::record_iterator testds_end = testds.end();
      for (typename dataset<la_type>::record_iterator testds_it(testds.begin());
           testds_it != testds_end; ++testds_it) {
        const record_type& example = *testds_it;
        if (predict(example) == label(example))
          ++test_acc;
      }
      test_acc /= testds.size();
      return test_acc;
    }

    //! Returns the accuracy on the given test data.
    //! @param n  max number of examples to be drawn from the given oracle
    value_type test_accuracy(oracle<la_type>& o, size_t n) const {
      size_t cnt(0);
      value_type test_acc(0);
      while (cnt < n) {
        if (!(o.next()))
          break;
        const record_type& example = o.current();
        if (predict(example) == label(example))
          ++test_acc;
        ++cnt;
      }
      if (cnt == 0) {
        std::cerr << "singlelabel_classifier::test_accuracy() called with an"
                  << " oracle from which no examples could be drawn."
                  << std::endl;
        assert(false);
        return -1;
      }
      test_acc /= cnt;
      return test_acc;
    }

    // Prediction methods
    //==========================================================================

    //! Predict the label of a new example.
    virtual std::size_t predict(const record_type& example) const = 0;

    virtual std::size_t predict(const assignment& example) const = 0;

    // Methods for iterative learners
    // (None of these are implemented by non-iterative learners.)
    //==========================================================================

    //! Returns the training accuracy after each iteration.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    virtual std::vector<value_type>
    train_accuracies() const {
      return std::vector<value_type>();
    }

    //! Computes the accuracy after each iteration on a test set.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    virtual std::vector<value_type>
    test_accuracies(const dataset<la_type>& testds) const {
      return std::vector<value_type>();
    }

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    virtual void save(std::ofstream& out, size_t save_part = 0,
                      bool save_name = true) const {
      base::save(out, save_part, save_name);
      out << label_index_ << "\n";
    }

    /**
     * Input the learner from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    virtual bool
    load(std::ifstream& in, const datasource& ds, size_t load_part) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> label_index_))
        assert(false);
      if (label_index_ < ds.num_finite())
        label_ = ds.finite_list()[label_index_];
      else {
        label_ = NULL;
        assert(false);
        return false;
      }
      return true;
    }

    // Protected data members
    //==========================================================================
  protected:

    //! Class variable
    finite_variable* label_;

    //! Index of class variable in dataset's records
    size_t label_index_;

  }; // class singlelabel_classifier

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_SINGLELABEL_CLASSIFIER_HPP
