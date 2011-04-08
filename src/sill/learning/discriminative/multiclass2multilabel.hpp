
#ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS2MULTILABEL_HPP
#define SILL_LEARNING_DISCRIMINATIVE_MULTICLASS2MULTILABEL_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>
#include <sill/learning/discriminative/multilabel_classifier.hpp>

#include <sill/macros_def.hpp>

// Set to true to print debugging information.
//#define DEBUG_MULTICLASS2MULTILABEL 0

namespace sill {

  struct multiclass2multilabel_parameters {

    //! Specifies base learner type
    //!  (required)
    boost::shared_ptr<multiclass_classifier<> > base_learner;

    //! New variable used to create a merged view of the class variables
    //!  (required)
    finite_variable* new_label;

    //! Used to make the algorithm, including the base classifier, deterministic
    //!  (default = time)
    double random_seed;

    multiclass2multilabel_parameters() : new_label(NULL) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    bool valid() const {
      if (base_learner.get() == NULL)
        return false;
      if (new_label == NULL)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      if (base_learner.get() == NULL) {
        std::cerr << "Error: cannot save multiclass2multilabel which does not"
                  << " have a base learner type." << std::endl;
        assert(false);
        return;
      }
      base_learner->save(out);
      out << random_seed << "\n";
    }

    void load(std::ifstream& in, const datasource& ds);

  }; // class multiclass2multilabel_parameters

  /**
   * Wrapper for converting a multiclass learner into a multilabel learner.
   *
   * Warning: This should only be used with small numbers of labels since its
   * time and space requirements grow exponentially with the number of labels!
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   */
  class multiclass2multilabel : public multilabel_classifier<> {

    // Public types
    //==========================================================================
  public:

    typedef multilabel_classifier<> base;

    typedef base::la_type la_type;
    typedef base::record_type record_type;

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    //  finite_var_vector labels_
    //  std::vector<size_t> label_indices_

    multiclass2multilabel_parameters params;

    //! Dataset view for converting records
    boost::shared_ptr<dataset_view<la_type> > ds_light_view;

    //! Base classifier
    boost::shared_ptr<multiclass_classifier<> > base_learner;

    //! Dataset structure for records which this classifier takes.
    datasource_info_type datasource_info_;

    //! Temp record for avoiding reallocation.
    mutable record_type tmp_rec;

    //! Temp assignment for avoiding reallocation.
    mutable assignment tmp_assign;

    // Protected methods
    //==========================================================================

    void init_only(const datasource& ds);

    void build(const dataset<la_type>& orig_ds);

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
    //  From multilabel_classifier:
    //   create()*
    //   predict()*
    //   confidences()
    //   marginal_probabilities()
    //   probabilities()

    // Constructors and destructors
    //==========================================================================

    /**
     * Constructor without associated data; useful for:
     *  - creating other instances
     *  - loading a saved booster
     * @param params        algorithm parameters
     */
    explicit multiclass2multilabel(multiclass2multilabel_parameters params
                                   = multiclass2multilabel_parameters())
      : base(), params(params) { }

    /**
     * Constructor.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit multiclass2multilabel(dataset_statistics<la_type>& stats,
                                   multiclass2multilabel_parameters params
                                   = multiclass2multilabel_parameters())
      : base(stats.get_dataset()), params(params),
        datasource_info_(stats.get_dataset().datasource_info()) {
      build(stats.get_dataset());
    }

    /**
     * Constructor.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    multiclass2multilabel(oracle<la_type>& o, size_t n,
                          multiclass2multilabel_parameters params
                          = multiclass2multilabel_parameters())
    : base(o), params(params), datasource_info_(o.datasource_info()) {
      boost::shared_ptr<vector_dataset<la_type> > ds_ptr(new vector_dataset<la_type>());
      oracle2dataset(o, n, *ds_ptr);
      build(*ds_ptr);
    }

    /**
     * Constructor which uses a pre-learned classifier.
     * @param ds  Datasource used for training.
     */
    multiclass2multilabel(boost::shared_ptr<multiclass_classifier<> > base_learner,
                          const datasource& ds,
                          multiclass2multilabel_parameters params
                          = multiclass2multilabel_parameters())
      : base(ds), params(params), base_learner(base_learner),
        datasource_info_(ds.datasource_info()) {
      init_only(ds);
    }

    //! Train a new multilabel classifier of this type with the given data.
    boost::shared_ptr<multilabel_classifier<> >
    create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<multilabel_classifier<> >
        bptr(new multiclass2multilabel(stats, this->params));
      return bptr;
    }

    //! Train a new multilabel classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<multilabel_classifier<> >
    create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<multilabel_classifier<> >
        bptr(new multiclass2multilabel(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const {
      return "multiclass2multilabel";
    }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name();
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const {
      return false;
    }

    //! Print classifier
    void print(std::ostream& out) const {
      out << "Multiclass2multilabel";
      if (base_learner)
        out << " with base:\n" << *base_learner;
      else
        out << " without base\n";
    }

    //! Returns the dataset structure for records which should be passed to this
    //! classifier.
    const datasource_info_type& datasource_info() const {
      return datasource_info_;
    }

    // Learning and mutating operations
    //==========================================================================

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    //! @todo Should this reset the base learner's random seed?
    void random_seed(double value) {
      params.random_seed = value;
//      rng.seed(static_cast<unsigned>(params.random_seed));
    }

    // Prediction methods
    //==========================================================================

    //! Predict the labels of a new example, storing the predictions in
    //! the given vector/assignment v.
    void predict(const record_type& example, std::vector<size_t>& v) const;

    void predict(const assignment& example, std::vector<size_t>& v) const;

    void predict(const record_type& example, finite_assignment& a) const;

    void predict(const assignment& example, finite_assignment& a) const;

    //! For each label, predict the marginal probability of each class.
    std::vector<vec> marginal_probabilities(const record_type& example) const;
    std::vector<vec> marginal_probabilities(const assignment& example) const;

    //! Returns a factor which gives the probability for each assignment
    //! to the labels.
    table_factor probabilities(const record_type& example) const;

    table_factor probabilities(const assignment& example) const;

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the classifier to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const;

    /**
     * Input the classifier from a human-readable file.
     * Note: This should only be called after the parameter NEW_LABEL has been
     *       set.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    bool load(std::ifstream& in, const datasource& ds, size_t load_part);

    // UNSAFE METHODS TO BE CHANGED LATER
    // =========================================================================

    boost::shared_ptr<multiclass_classifier<> > get_base_learner_ptr() const {
      return base_learner;
    }

    //! Sets the given record to be the proper size for passing to
    //! convert_record_for_base().
    void prepare_record_for_base(record_type& new_r) const {
      new_r = tmp_rec;
    }

    //! Converts the original record (in the data format used by this class)
    //! to a new one (in the data format used by this class' base learner.
    //! @todo Figure out a better way to do this; this is needed by
    //!       log_reg_crf_factor.
    void convert_record_for_base(const record_type& orig_r, record_type& new_r) const {
      ds_light_view->convert_record(orig_r, new_r);
    }

  }; // class multiclass2multilabel

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS2MULTILABEL_HPP
