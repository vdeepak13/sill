#ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS2MULTILABEL_HPP
#define SILL_LEARNING_DISCRIMINATIVE_MULTICLASS2MULTILABEL_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/learning/dataset_old/data_conversions.hpp>
#include <sill/learning/dataset_old/dataset_view.hpp>
#include <sill/learning/dataset_old/vector_dataset.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>
#include <sill/learning/discriminative/multilabel_classifier.hpp>

#include <sill/macros_def.hpp>

// Set to true to print debugging information.
//#define DEBUG_MULTICLASS2MULTILABEL 0

namespace sill {

  // Forward declaration
  template <typename LA>
  boost::shared_ptr<multiclass_classifier<LA> >
  load_multiclass_classifier(std::ifstream& in, const datasource& ds);


  //! Parameters for multiclass2multilabel.
  template <typename LA = dense_linear_algebra<> >
  struct multiclass2multilabel_parameters {

    typedef LA la_type;

    //! Specifies base learner type
    //!  (required)
    boost::shared_ptr<multiclass_classifier<la_type> > base_learner;

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

    void load(std::ifstream& in, const datasource& ds) {
      base_learner = load_multiclass_classifier<la_type>(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> random_seed))
        assert(false);
    }

  }; // class multiclass2multilabel_parameters

  /**
   * Wrapper for converting a multiclass learner into a multilabel learner.
   *
   * Warning: This should only be used with small numbers of labels since its
   * time and space requirements grow exponentially with the number of labels!
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   */
  template <typename LA = dense_linear_algebra<> >
  class multiclass2multilabel : public multilabel_classifier<LA> {

    // Public types
    //==========================================================================
  public:

    typedef LA la_type;

    typedef multilabel_classifier<la_type> base;

    typedef typename base::record_type record_type;

    typedef multiclass2multilabel_parameters<la_type> parameters;

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
    explicit multiclass2multilabel(parameters params = parameters())
      : base(), params(params) { }

    /**
     * Constructor.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit multiclass2multilabel(dataset_statistics<la_type>& stats,
                                   parameters params = parameters())
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
                          parameters params = parameters())
    : base(o), params(params), datasource_info_(o.datasource_info()) {

      boost::shared_ptr<vector_dataset_old<la_type> >
        ds_ptr(new vector_dataset_old<la_type>());
      oracle2dataset(o, n, *ds_ptr);
      build(*ds_ptr);
    }

    /**
     * Constructor which uses a pre-learned classifier.
     * @param ds  Datasource used for training.
     */
    multiclass2multilabel
    (boost::shared_ptr<multiclass_classifier<la_type> > base_learner,
     const datasource& ds,
     parameters params = parameters())
      : base(ds), params(params), base_learner(base_learner),
        datasource_info_(ds.datasource_info()) {
      init_only(ds);
    }

    //! Train a new multilabel classifier of this type with the given data.
    boost::shared_ptr<multilabel_classifier<la_type> >
    create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<multilabel_classifier<la_type> >
        bptr(new multiclass2multilabel(stats, this->params));
      return bptr;
    }

    //! Train a new multilabel classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<multilabel_classifier<la_type> >
    create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<multilabel_classifier<la_type> >
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

    // Methods for iterative learners
    //==========================================================================

    //! Call this after learning to free memory.
    //! NOTE: Once this method has been called, step() may fail!
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void finish_learning() {
      if (base_learner)
        base_learner->finish_learning();
    }

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

    boost::shared_ptr<multiclass_classifier<la_type> >
    get_base_learner_ptr() const {
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
    void convert_record_for_base(const record_type& orig_r,
                                 record_type& new_r) const {
      if (ds_light_view)
        ds_light_view->convert_record(orig_r, new_r);
      else
        new_r = orig_r;
    }

    // Protected data members
    //==========================================================================
  protected:

    // Import from base class:
    using base::labels_;
    using base::label_indices_;

    parameters params;

    //! Dataset view for converting records.
    //! If NULL, then there is only a single label, so no conversion is needed.
    boost::shared_ptr<dataset_view<la_type> > ds_light_view;

    //! Base classifier
    boost::shared_ptr<multiclass_classifier<la_type> > base_learner;

    //! Dataset structure for records which this classifier takes.
    datasource_info_type datasource_info_;

    //! Temp record for avoiding reallocation.
    mutable record_type tmp_rec;

    //! Temp assignment for avoiding reallocation.
    mutable assignment tmp_assign;

    // Protected methods
    //==========================================================================

    void init_sub() {
      assert(params.valid());
      size_t new_label_size = num_assignments(make_domain(labels_));
      if (params.new_label == NULL ||
          new_label_size != params.new_label->size()) {
        assert(false);
        return;
      }
    }

    void init_only(const datasource& ds) {
      if (ds.finite_class_variables().size() > 1) {
        init_sub();
        vector_dataset_old<la_type> orig_ds(ds.datasource_info());
        dataset_view<la_type> ds_view(orig_ds, true);
        ds_view.set_merged_variables(orig_ds.finite_class_variables(),
                                     params.new_label);
        ds_light_view = ds_view.create_light_view();
        tmp_rec = ds_view[0];
      } else {
        assert(ds.finite_class_variables().size() == 1);
        ds_light_view.reset();
      }

      params.base_learner = base_learner;
    }

    void build(const dataset<la_type>& orig_ds) {
      init_sub();

      boost::mt11213b rng(static_cast<unsigned>(params.random_seed));
      params.base_learner->random_seed
        (boost::uniform_int<int>(0, std::numeric_limits<int>::max())(rng));

      if (orig_ds.finite_class_variables().size() > 1) {
        dataset_view<la_type> ds_view(orig_ds, true);
        ds_view.set_merged_variables(orig_ds.finite_class_variables(),
                                     params.new_label);
        dataset_statistics<la_type> stats(ds_view);
        ds_light_view = ds_view.create_light_view();
        tmp_rec = ds_view[0];

        base_learner = params.base_learner->create(stats);
      } else {
        assert(orig_ds.finite_class_variables().size() == 1);
        ds_light_view.reset();
        dataset_statistics<la_type> stats(orig_ds);

        base_learner = params.base_learner->create(stats);
      }
    }

    std::vector<vec> probs2marginals(const table_factor& probs) const {
      std::vector<vec> v(labels_.size());
      for (size_t j(0); j < labels_.size(); ++j) {
        v[j].set_size(labels_[j]->size());
        size_t j2(0);
        foreach(double val, probs.marginal(make_domain(labels_[j])).values()) {
          v[j][j2] = val;
          ++j2;
        }
      }
      return v;
    }

  }; // class multiclass2multilabel


  //============================================================================
  // Implementations of methods in multiclass2multilabel
  //============================================================================

  // Prediction methods
  //==========================================================================

  template <typename LA>
  void
  multiclass2multilabel<LA>::
  predict(const record_type& example, std::vector<size_t>& v) const {
    if (ds_light_view) {
      ds_light_view->convert_record(example, tmp_rec);
      size_t pred = base_learner->predict(tmp_rec);
      ds_light_view->revert_merged_value(pred, v);
    } else {
      v.resize(1);
      v[0] = base_learner->predict(example);
    }
  }

  template <typename LA>
  void
  multiclass2multilabel<LA>::
  predict(const assignment& example, std::vector<size_t>& v) const {
    if (ds_light_view) {
      ds_light_view->convert_assignment(example, tmp_assign);
      size_t pred = base_learner->predict(tmp_assign);
      ds_light_view->revert_merged_value(pred, v);
    } else {
      v.resize(1);
      v[0] = base_learner->predict(example);
    }
  }

  template <typename LA>
  void
  multiclass2multilabel<LA>::
  predict(const record_type& example, finite_assignment& a) const {
    if (ds_light_view) {
      ds_light_view->convert_record(example, tmp_rec);
      ds_light_view->revert_merged_value(base_learner->predict(tmp_rec), a);
    } else {
      assert(labels_.size() == 1);
      a[labels_[0]] = base_learner->predict(example);
    }
  }

  template <typename LA>
  void
  multiclass2multilabel<LA>::
  predict(const assignment& example, finite_assignment& a) const {
    if (ds_light_view) {
      ds_light_view->convert_assignment(example, tmp_assign);
      ds_light_view->revert_merged_value(base_learner->predict(tmp_assign), a);
    } else {
      assert(labels_.size() == 1);
      a[labels_[0]] = base_learner->predict(example);
    }
  }

  template <typename LA>
  std::vector<vec>
  multiclass2multilabel<LA>::
  marginal_probabilities(const record_type& example) const {
    table_factor probs(probabilities(example));
    return probs2marginals(probs);
  }

  template <typename LA>
  std::vector<vec>
  multiclass2multilabel<LA>::
  marginal_probabilities(const assignment& example) const {
    table_factor probs(probabilities(example));
    return probs2marginals(probs);
  }

  template <typename LA>
  table_factor
  multiclass2multilabel<LA>::probabilities(const record_type& example) const {
    if (ds_light_view) {
      ds_light_view->convert_record(example, tmp_rec);
      return make_dense_table_factor(labels_,
                                     base_learner->probabilities(tmp_rec));
    } else {
      return make_dense_table_factor(labels_,
                                     base_learner->probabilities(example));
    }
  }

  template <typename LA>
  table_factor
  multiclass2multilabel<LA>::probabilities(const assignment& example) const {
    if (ds_light_view) {
      ds_light_view->convert_assignment(example, tmp_assign);
      return make_dense_table_factor(labels_,
                                     base_learner->probabilities(tmp_assign));
    } else {
      return make_dense_table_factor(labels_,
                                     base_learner->probabilities(example));
    }
  }

  // Save and load methods
  //==========================================================================

  template <typename LA>
  void multiclass2multilabel<LA>::save(std::ofstream& out, size_t save_part,
                                       bool save_name) const {
    base::save(out, save_part, save_name);
    params.save(out);
    if (ds_light_view) {
      out << true << "\n";
      ds_light_view->save(out);
    } else {
      out << false << "\n";
    }
    base_learner->save(out, 0, true);
  }

  template <typename LA>
  bool multiclass2multilabel<LA>::load(std::ifstream& in, const datasource& ds,
                                       size_t load_part) {
    if (!(base::load(in, ds, load_part)))
      return false;
    params.load(in, ds);
    bool load_ds_light_view;
    in >> load_ds_light_view;
    in >> std::ws;
    if (load_ds_light_view)
      ds_light_view->load(in, NULL, params.new_label);
    else
      ds_light_view.reset();
    base_learner = load_multiclass_classifier<la_type>(in, ds);
    tmp_rec = record_type(ds.finite_numbering_ptr(), ds.vector_numbering_ptr(),
                          ds.vector_dim());
    return true;
  }


} // namespace sill

#include <sill/macros_undef.hpp>

#include <sill/learning/discriminative/load_functions.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS2MULTILABEL_HPP
