
#ifndef SILL_DATASET_VIEW_HPP
#define SILL_DATASET_VIEW_HPP

#include <sill/learning/dataset/dataset.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  template <typename LA> class vector_dataset;

  /**
   * A class that provides a view of a subset of another dataset.
   * This currently supports views which may be defined by any or all of these:
   *  - Record views: range of record indices or a set of indices.
   *     (efficient)
   *  - Variable views: set of indices.
   *  - Binarization of a finite variable (making any finite variable appear to
   *    be binary-valued according to a given coloring).
   *     (efficient)
   *  - Merging of several finite variables into a new finite variable
   *    (of arity equal to the product of the original arities).
   *     (not too efficient)
   *  - TODO: Merging multiple datasets (which have the same variable orderings)
   * Usage notes:
   *  - Binarization and merging can only be done once; to binarize
   *    more than one variable or to merge more than one set of variables,
   *    you must construct a view, cast it to a dataset pointer/reference,
   *    and then construct another view from that.  At that point, it might
   *    become more efficient to use the view to generate a new dataset
   *    (copying the modified data from the view into an actual dataset).
   *    Also, a binarized variable may not be merged, but a new variable
   *    created by merging variables may be binarized.
   *
   * Implementation notes:
   *  - Views may not be undone; i.e., if a view has been created over
   *    a restricted set of records, then it cannot be used to create a view
   *    over all of the records.
   *  - A dataset_view may be constructed from a dataset or a dataset_view.
   *    If the dataset is really a dataset_view, then the resulting dataset will
   *    be more efficient if the dataset_view is constructed from a dataset_view
   *    instead of a dataset.
   *  - This does not have an assignment operator.  Just make a new view.
   *    (They are relatively light objects.)
   *  - Records hold finite_numbering and vector_numbering, which specify the
   *    order
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   * \todo serialization
   * \todo This should probably be changed to hold a copy_ptr to the data
   *       so that the underlying data can be modified safely.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<double,size_t>)
   */
  template <typename LA = dense_linear_algebra<> >
  class dataset_view : public dataset<LA> {

    // Public type declarations
    //==========================================================================
  public:

    //! Base class
    typedef dataset<LA> base;

    //! Import stuff from base class
    typedef typename base::record_type record_type;
    typedef typename base::vector_type vector_type;
    typedef typename base::record_iterator_type record_iterator_type;

    // Constructors
    //==========================================================================
  public:

    /**
     * Construct a view of a dataset which is the same as the original dataset.
     * @param keep_weights  if true, then records keep their associated weights
     * @todo keep_weights should be true by default; see if changing this will
     *       mess anything up.
     */
    explicit dataset_view(const dataset_view& ds_view,
                          bool keep_weights = false)
      : base(ds_view), ds_ptr(ds_view.ds_ptr), ds(ds_view.ds),
        record_view(ds_view.record_view),
        record_min(ds_view.record_min), record_max(ds_view.record_max),
        record_indices(ds_view.record_indices), saved_record_indices(NULL),
        vv_view(ds_view.vv_view),
        vv_finite_var_indices(ds_view.vv_finite_var_indices),
        vv_vector_var_indices(ds_view.vv_vector_var_indices),
        binarized_var(ds_view.binarized_var),
        binarized_var_index(ds_view.binarized_var_index),
        binary_var(ds_view.binary_var),
        binary_coloring(ds_view.binary_coloring),
        m_new_var(ds_view.m_new_var), m_new_var_index(ds_view.m_new_var_index),
        m_orig_vars(ds_view.m_orig_vars),
        m_orig_vars_indices(ds_view.m_orig_vars_indices),
        m_multipliers(ds_view.m_multipliers),
        m_orig_vars_sorted(ds_view.m_orig_vars_sorted),
        m_multipliers_sorted(ds_view.m_multipliers_sorted),
        m_orig2new_indices(ds_view.m_orig2new_indices),
        m_new2orig_indices(ds_view.m_new2orig_indices),
        tmp_findata(ds_view.tmp_findata), tmp_vecdata(ds_view.tmp_vecdata) {
      if (!keep_weights) {
        weighted = false;
        weights_.clear();
      }
      if (ds_view.saved_record_indices)
        saved_record_indices =
          new std::vector<size_t>(*(ds_view.saved_record_indices));
    }

    /**
     * Construct a view of a dataset which is the same as the original dataset.
     * @param keep_weights  if true, then records keep their associated weights
     */
    explicit dataset_view(const dataset<LA>& ds, bool keep_weights = false)
      : base(ds), ds_ptr(NULL), ds(ds),
        record_view(RECORD_ALL), saved_record_indices(NULL),
        vv_view(VAR_ALL), binarized_var(NULL), m_new_var(NULL) {
      if (keep_weights && ds.is_weighted()) {
        weighted = true;
        weights_.resize(nrecords);
        for (size_t i = 0; i < nrecords; ++i)
          weights_[i] = ds.weight(convert_index(i));
      } else {
        weighted = false;
        weights_.clear();
      }
    }

    ~dataset_view() {
      if (ds_ptr != NULL)
        delete(ds_ptr);
      if (saved_record_indices)
        delete(saved_record_indices);
    }

    //! Assignment operator.
    dataset_view& operator=(const dataset_view& ds_view) {
      assert(false); // TO DO
      return *this;
    }

    // Getters and helpers
    //==========================================================================

    // From datasource
    using base::has_variable;
    using base::num_finite;
    using base::num_vector;
    using base::variable_type_order;
    using base::var_order;
    //    using base::var_order_index;
    //    using base::variable_index;
    using base::record_index;
    using base::vector_indices;
    using base::finite_numbering;
    using base::finite_numbering_ptr;
    using base::vector_numbering;
    using base::vector_numbering_ptr;

    // From dataset
    using base::size;
    using base::weight;

    //! Return capacity
    size_t capacity() const { return ds.capacity(); }

    //! Element access: record i, finite variable j (in the order finite_list())
    //! Note: Full record retrievals are more efficient than this function.
    size_t finite(size_t i, size_t j) const;

    using base::vector;

    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! Note: Full record retrievals are more efficient than this function.
    double vector(size_t i, size_t j) const;

    //! Returns an iterator over the records of this dataset
    record_iterator_type begin() const {
      if (m_new_var == NULL && vv_view == VAR_ALL && binarized_var == NULL) {
        record_iterator_type it(ds.begin());
        it.data = this;
        return it;
      } else {
        return make_record_iterator(0);
      }
    }

    //! Convert an assignment from the original dataset into one for this view.
    //! Note: The caller must know what the original dataset looked like!
    void convert_assignment(const assignment& orig_r, assignment& new_r) const;

    //! Convert a record from the original dataset into one for this view.
    //! Note: The caller must know what the original dataset looked like!
    //! @param new_r  Pre-allocated and properly sized record.
    void convert_record(const record_type& orig_r, record_type& new_r) const;

    //! (When using merged variables:)
    //! Convert a value for the (new) merged variable into a list of values
    //! for the (original) merged variables.
    //! @param orig_vals  The original values are stored here in the order
    //!                   used by finite variables in the original dataset.
    void revert_merged_value(size_t merged_val,
                             std::vector<size_t>& orig_vals) const;

    //! (When using merged variables:)
    //! Convert a value for the (new) merged variable into a list of values
    //! for the (original) merged variables.
    //! @param orig_vals  The original values are stored here in the order
    //!                   used by finite variables in the original dataset.
    void revert_merged_value(size_t merged_val,
                             finite_assignment& orig_vals) const;

    //! Create a light view (without data) which
    //! can be used to compute views of new records and assignments.
    //! @see convert_assignment, convert_record
    boost::shared_ptr<dataset_view> create_light_view() const;

    // Mutating operations: creating views
    //==========================================================================

    /**
     * Restrict this view to the range of records [min, max).
     */
    void set_record_range(size_t min, size_t max);

    /**
     * Restrict this view to records with the given indices.
     * The indices' order specifies the new order of the records.
     * Note that multiple copies of an index will simulate multiple copies
     * of a record.
     */
    void set_record_indices(const std::vector<size_t>& indices);

    /**
     * Restricts this view to be fold 'fold' out of 'nfolds' for n-fold
     * cross-validation. This is meant to be used by creating a new view
     * for each fold.
     *
     * Note: If the dataset size S is not an integer multiple of nfolds,
     *       then the held-out fold i will have records
     *       [floor(i * S / nfolds), floor((i+1) * S / nfolds)-1].
     *
     * @param fold     Fold number (0 to nfolds-1).
     * @param nfolds   Number of cross-validation folds
     *                 (1 < nfolds <= dataset view size).
     * @param heldout  If true, this returns the held-out fold;
     *                 if false, this returns the rest of the dataset.
     */
    void set_cross_validation_fold(size_t fold, size_t nfolds, bool heldout);

    /**
     * This saves the current record view, which can be restored later using
     * restore_record_view().
     */
    void save_record_view();

    /**
     * This restores the last record view saved by save_record_view().
     * The saved record view remains saved.
     * If save_record_view() has not been called, this asserts false.
     */
    void restore_record_view();

    /**
     * Change this view so that only a subset of the variables are visible.
     * Note: The order of the new set of variables will be a suborder of the
     *       original order.
     * Warning: This is not yet compatible with binarized or merged variables!
     */
    void set_variable_indices(const std::set<size_t>& finite_indices,
                              const std::set<size_t>& vector_indices);

    /**
     * Change this view so that only a subset of the variables are visible.
     * Note: The order of the new set of variables will be a suborder of the
     *       original order.
     * Warning: This is not yet compatible with binarized or merged variables!
     */
    void set_variables(const finite_domain& fvars, const vector_domain& vvars);

    /**
     * Change this view so that it changes the original variable to a binary
     * variable by making its value 0/1 based on the given coloring.
     *
     * Note: This only supports binarization of a single variable.
     *
     * @param original  finite variable in the dataset
     * @param binary    binary finite variable not in the dataset
     * @param coloring  coloring[j] = 0/1 value to replace value j of original
     */
    void set_binary_coloring(finite_variable* original, finite_variable* binary,
                             std::vector<size_t> coloring);

    /**
     * Change this view so that it changes the original variable to a binary
     * variable by making its value 1 if the original value was 'val' and 0
     * otherwise.
     *
     * Note: This only supports binarization of a single variable.
     *
     * @param original  finite variable in the dataset
     * @param binary    binary finite variable not in the dataset
     * @param one_val   value of the original variable
     */
    void set_binary_indicator(finite_variable* original,
                              finite_variable* binary, size_t one_val);

    /**
     * Change this view so that it merges multiple finite variables into a
     * single new finite variable of arity equal to the product of the arities
     * of the merged variables.
     *
     * Note: The merged variables must either all be class variables
     *       (in which case the new variable will also be a class variable)
     *       or all be non-class variables (in which case the new variable
     *       will be a non-class variable).
     *
     * Variable ordering: This uses the ordering of original_vars, which does
     *     not have to coincide with the dataset variable ordering.
     *     This computes the new variable's values using the right-most
     *     variable in original_vars as the most significant digit;
     *     this is the same as the make_dense_table_factor() function.
     *
     * @param  original_vars  original variables (to be merged) (size > 1)
     * @param  new_var        new finite variable (of proper arity)
     */
    void set_merged_variables(finite_var_vector original_vars,
                              finite_variable* new_var);

    /**
     * Change this view so that it only includes records with the given values.
     * This has the same qualifications as set_record_indices().
     * Note: This does not change the variables in the dataset (which is a good
     *       idea if learners expect fixed variable orderings).
     * @param fa  Only records with these values are kept.
     *            The dataset must include all variables in this assignment.
     */
    void restrict_to_assignment(const finite_assignment& fa);

    // Mutating operations: datasets
    //==========================================================================

    //! Increases the capacity in anticipation of adding new elements.
    void reserve(size_t n) {
      assert(false);
    }

    //! Adds a new record with weight w (default = 1)
    //! \todo Add support for this.  If we allow dataset views of merged
    //!       datasets, we could similarly have dataset_view contain an extra
    //!       dataset into which new records could be inserted.
    void insert(const assignment& a, double w = 1) {
      assert(false);
    }

    //! Adds a new record with weight w (default = 1)
    void insert(const record_type& r, double w = 1) {
      assert(false);
    }

    //! Adds a new record with finite variable values fvals and vector variable
    //! values vvals, with weight w (default = 1).
    void insert(const std::vector<size_t>& fvals, const vector_type& vvals,
                double w = 1) {
      assert(false);
    }

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const assignment& a, double w = 1) {
      assert(false);
    }

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const std::vector<size_t>& fvals,
                    const vector_type& vvals, double w = 1) {
      assert(false);
    }

    //! Normalizes the vector data using the given means and std_devs
    //!  (which are assumed to be correct).
    void normalize(const vec& means, const vec& std_devs,
                   const vector_var_vector& vars) {
      assert(false);
    }

    //! Normalizes the vector data so that each record's vector values lie
    //! on the unit sphere.
    //! @param vars  Only apply normalization to these variables.
    void normalize2(const vector_var_vector& vars) {
      assert(false);
    }

    //! Randomly reorders the dataset view (this is a mutable operation)
    void randomize(double random_seed);

    // Save and load methods
    //==========================================================================

    //! Output the dataset view to a human-readable file which can be reloaded.
    //! This only saves the light view (without the data).
    void save(std::ofstream& out) const;

    //! Output the dataset view to a human-readable file which can be reloaded.
    //! This only saves the light view (without the data).
    void save(const std::string& filename) const;

    /**
     * Input the dataset view from a human-readable file.
     * This view must have been created with the original dataset,
     * or an empty dataset with the same datasource properties
     * (for a light view).
     * @param in          input filestream for file holding the saved view
     * @param binary_var_ same as binary var passed to set_binary_coloring()
     * @param m_new_var_  same as finite var passed to set_merged_variables()
     * @return  true if successful
     */
    bool load(std::ifstream& in, finite_variable* binary_var_ = NULL,
              finite_variable* m_new_var_ = NULL);

    /**
     * Input the dataset view from a human-readable file.
     * This view must have been created with the original dataset,
     * or an empty dataset with the same datasource properties
     * (for a light view).
     * @param filename    filename of saved view
     * @param binary_var_ same as binary var passed to set_binary_coloring()
     * @param m_new_var_  same as finite var passed to set_merged_variables()
     * @return  true if successful
     */
    bool load(const std::string& filename, finite_variable* binary_var_ = NULL,
              finite_variable* m_new_var_ = NULL);

    // Protected data members
    //==========================================================================
  protected:

    // From datasource
    //    using base::finite_vars;
    using base::finite_seq;
    using base::finite_numbering_ptr_;
    using base::dfinite;
    using base::finite_class_vars;
    //    using base::vector_vars;
    using base::vector_seq;
    using base::vector_numbering_ptr_;
    using base::dvector;
    using base::vector_class_vars;
    using base::var_type_order;
    //    using base::var_order_map;
    //    using base::vector_var_order_map;

    // From dataset
    using base::nrecords;
    using base::weighted;
    using base::weights_;
    using base::load_record;
    using base::load_finite;
    using base::load_vector;
    using base::make_record_iterator;

    //! Used for deleting the dataset to make a light class which can convert
    //! records according to the view.
    vector_dataset<LA>* ds_ptr;

    //! Dataset
    const dataset<LA>& ds;

    // Protected data members: for record views
    //==========================================================================

    //! Types of views for records
    enum record_view_type {RECORD_RANGE, RECORD_INDICES, RECORD_ALL};

    //! Type of view for records
    record_view_type record_view;

    //! RECORD_RANGE: min (included)
    size_t record_min;

    //! RECORD_RANGE: max (excluded)
    size_t record_max;

    //! RECORD_INDICES
    std::vector<size_t> record_indices;

    //! For SAVE_RECORD_VIEW() and RESTORE_RECORD_VIEW()
    //! If NULL, then this is not in use.
    std::vector<size_t>* saved_record_indices;

    // Protected data members: for variable views
    //==========================================================================

    //! Types of views for variables
    enum var_view_type {VAR_INDICES, VAR_ALL};

    //! Type of view for variables
    var_view_type vv_view;

    //! Indices of finite variables in view in original dataset's finite data
    std::vector<size_t> vv_finite_var_indices;

    //! Indices of vector variables in view in original dataset's vector data
    ivec vv_vector_var_indices;
    //    std::vector<size_t> vv_vector_var_indices;

    // Protected data members: for binarizing finite variables
    //==========================================================================

    //! Finite variable which is being viewed as binary.
    //! If null, then no binarized variable.
    finite_variable* binarized_var;

    //! Index of binarized variable in records' finite data.
    size_t binarized_var_index;

    //! Binary variable used to replace the binarized variable.
    finite_variable* binary_var;

    //! Value of finite variable which has value 1 in the binary variable.
    //! All others are 0.
    //    size_t binarized_value;
    //! Boolean 0/1 vector: binary_coloring[j] = 0/1 for label j
    std::vector<size_t> binary_coloring;

    // Protected data members: for merging finite variables
    //==========================================================================

    // Note: Merging is done before binarizing, so when merging is set,
    //       it affects the indices for binarizing.
    //! Finite variable used to replace the merged variables.
    //! If null, then no merged variables.
    finite_variable* m_new_var;

    //! Index of m_new_var in (new) records' finite data.
    size_t m_new_var_index;

    //! Old (merged) finite variables.
    finite_var_vector m_orig_vars;

    //! Indices of m_orig_vars in (old) records' finite data.
    std::vector<size_t> m_orig_vars_indices;

    //! Multipliers used to compute the value of new_merged_var from the values
    //! of m_orig_vars:
    //!  new val = sum_i m_multipliers_i x merged_var_value_i,
    //!   where merged vars are indexed in the order given by m_orig2new_indices
    //! NOTE: This must work the same way as make_dense_table_factor().
    std::vector<size_t> m_multipliers;

    //! Version sorted in the original order (so m_multipliers is sorted).
    finite_var_vector m_orig_vars_sorted;

    //! Version sorted in the original order (so m_multipliers is sorted).
    std::vector<size_t> m_multipliers_sorted;

    //! m_orig2new_indices[j]
    //!    = index of non-merged finite var j (in original records)
    //!       in the new records
    //!   or std::numeric_limits<size_t>::max() if j is merged
    std::vector<size_t> m_orig2new_indices;

    //! m_new2orig_indices[j]
    //!    = index of non-merged finite var j (in the new records)
    //!       in the original records
    //!   or std::numeric_limits<size_t>::max() if j == m_new_var_index
    std::vector<size_t> m_new2orig_indices;

    //! Temp findata (of original size) for avoiding repeated allocation.
    //! Used for variable views and merged variables.
    mutable std::vector<size_t> tmp_findata;

    //! Temp vecdata (of original size) for avoiding repeated allocation.
    //! Used for variable views.
    mutable vector_type tmp_vecdata;

    // Protected functions
    //==========================================================================

    //! Translates index i (into view) into index for actual dataset.
    size_t convert_index(size_t i) const;

    //! Constructor for a view without associated data;
    //! used by create_light_view().
    dataset_view(vector_dataset<LA>* ds_ptr, const dataset_view& ds_view_source)
      : base(ds_view_source), ds_ptr(ds_ptr), ds(*ds_ptr),
        record_view(RECORD_ALL), saved_record_indices(NULL),
        vv_view(VAR_ALL), binarized_var(NULL), m_new_var(NULL) { }

    // Protected functions required by record
    //==========================================================================

    //! Used by load_assignment() and convert_assignment()
    void convert_assignment_(assignment& a) const;

    //! Load datapoint i into assignment a
    void load_assignment(size_t i, sill::assignment& a) const;

    //! Load record i into r
    void load_record(size_t i, record_type& r) const;

    //! Load finite data for datapoint i into findata
    void load_finite(size_t i, std::vector<size_t>& findata) const;

    //! Load vector data for datapoint i into vecdata
    void load_vector(size_t i, vector_type& vecdata) const;

    //! ONLY for datasets which use assignments as native types:
    //!  Load the pointer to datapoint i into (*a).
    void load_assignment_pointer(size_t i, assignment** a) const;

  };  // class dataset_view

  //============================================================================
  // Implementations of methods in dataset_view
  //============================================================================

  // Protected functions
  //==========================================================================

  template <typename LA>
  size_t dataset_view<LA>::convert_index(size_t i) const {
    switch(record_view) {
    case RECORD_RANGE:
      return record_min + i;
    case RECORD_INDICES:
      return record_indices[i];
    case RECORD_ALL:
      return i;
    default:
      assert(false);
      return 0;
    }
  }

  // Protected functions required by record
  //==========================================================================

  template <typename LA>
  void dataset_view<LA>::convert_assignment_(assignment& a) const {
    finite_assignment& fa = a.finite();
    if (binarized_var != NULL) {
      fa[binary_var] =
        binary_coloring[fa[binarized_var]];
      fa.erase(binarized_var);
    }
    if (m_new_var != NULL) {
      size_t val(0);
      for (size_t j(0); j < m_orig_vars.size(); ++j)
        val += m_multipliers[j] * fa[m_orig_vars[j]];
      fa[m_new_var] = val;
      for (size_t j(0); j < m_orig_vars.size(); ++j)
        fa.erase(m_orig_vars[j]);
    }
    if (vv_view != VAR_ALL) {
      size_t j2(0); // index into vv_finite_var_indices
      for (size_t j(0); j < ds.num_finite(); ++j) {
        if (j2 < vv_finite_var_indices.size() &&
            vv_finite_var_indices[j2] == j) {
          ++j2;
          continue;
        }
        fa.erase(ds.finite_list()[j]);
      }
      vector_assignment& va = a.vector();
      j2 = 0; // index into vv_vector_var_indices
      for (size_t j(0); j < ds.num_vector(); ++j) {
        if (j2 < vv_vector_var_indices.size() &&
            (size_t)(vv_vector_var_indices[j2]) == j) {
          ++j2;
          continue;
        }
        va.erase(ds.vector_list()[j]);
      }
    }
  }

  template <typename LA>
  void dataset_view<LA>::load_assignment(size_t i, sill::assignment& a) const {
    ds.load_assignment(convert_index(i), a);
    convert_assignment_(a);
  }

  template <typename LA>
  void dataset_view<LA>::load_record(size_t i, record_type& r) const {
    if (m_new_var != NULL) {
      ds.load_vector(convert_index(i), r.vector());
      ds.load_finite(convert_index(i), tmp_findata);
      // Note: tmp_findata is larger than r.finite() is.
      std::vector<size_t>& fin = r.finite();
      size_t val(0);
      size_t j2(0); // index into m_multipliers
      for (size_t j(0); j < m_orig2new_indices.size(); ++j) {
        if (m_orig2new_indices[j] == std::numeric_limits<size_t>::max()) {
          val += m_multipliers[j2] * tmp_findata[j];
          ++j2;
        } else
          fin[m_orig2new_indices[j]] = tmp_findata[j];
      }
      fin[m_new_var_index] = val;
    } else if (vv_view != VAR_ALL) { // TO DO: SPEED THIS UP!
      ds.load_vector(convert_index(i), tmp_vecdata);
      ds.load_finite(convert_index(i), tmp_findata);
      std::vector<size_t>& fin = r.finite();
      for (size_t j(0); j < vv_finite_var_indices.size(); ++j)
        fin[j] = tmp_findata[vv_finite_var_indices[j]];
      r.vector() = tmp_vecdata(vv_vector_var_indices);
      //        vector_type& v = r.vector();
      //        for (size_t j(0); j < vv_vector_var_indices.size(); ++j)
      //          v[j] = tmp_vecdata[vv_vector_var_indices[j]];
    } else {
      ds.load_record(convert_index(i), r);
    }
    if (binarized_var != NULL) // move this to below if-then
      r.finite(binarized_var_index) =
        binary_coloring[r.finite(binarized_var_index)];
    r.finite_numbering_ptr = finite_numbering_ptr_;
    r.vector_numbering_ptr = vector_numbering_ptr_;
  }

  template <typename LA>
  void
  dataset_view<LA>::load_finite(size_t i, std::vector<size_t>& findata) const {
    if (m_new_var != NULL) {
      ds.load_finite(convert_index(i), tmp_findata);
      // Note: tmp_findata is larger than findata is.
      size_t val(0);
      size_t j2(0); // index into m_multipliers
      for (size_t j(0); j < m_orig2new_indices.size(); ++j) {
        if (m_orig2new_indices[j] ==std::numeric_limits<size_t>::max()){
          val += m_multipliers[j2] * tmp_findata[j];
          ++j2;
        } else
          findata[m_orig2new_indices[j]] = tmp_findata[j];
      }
      findata[m_new_var_index] = val;
    } else if (vv_view != VAR_ALL) {
      ds.load_finite(convert_index(i), tmp_findata);
      for (size_t j(0); j < vv_finite_var_indices.size(); ++j)
        findata[j] = tmp_findata[vv_finite_var_indices[j]];
    } else {
      ds.load_finite(convert_index(i), findata);
    }
    if (binarized_var != NULL) {
      findata[binarized_var_index] =
        binary_coloring[findata[binarized_var_index]];
    }
  }

  template <typename LA>
  void dataset_view<LA>::load_vector(size_t i, vector_type& vecdata) const {
    if (vv_view != VAR_ALL) {
      ds.load_vector(convert_index(i), tmp_vecdata);
      vecdata = tmp_vecdata(vv_vector_var_indices);
      //        for (size_t j(0); j < vv_vector_var_indices.size(); ++j)
      //          vecdata[j] = tmp_vecdata[vv_vector_var_indices[j]];
    } else
      ds.load_vector(convert_index(i), vecdata);
  }

  template <typename LA>
  void dataset_view<LA>::load_assignment_pointer(size_t i, assignment** a) const {
    assert(false);
  }

  // Getters and helpers
  //==========================================================================

  template <typename LA>
  size_t dataset_view<LA>::finite(size_t i, size_t j) const {
    size_t val(0);
    if (m_new_var == NULL)
      val = ds.finite(convert_index(i), j);
    else if (vv_view != VAR_ALL) {
      if (j >= finite_seq.size()) {
        assert(false);
        return 0;
      }
      val = ds.finite(convert_index(i), vv_finite_var_indices[j]);
    } else {
      if (j >= finite_seq.size()) {
        assert(false);
        return 0;
      }
      if (j != m_new_var_index)
        val = ds.finite(convert_index(i), m_new2orig_indices[j]);
      else
        for (size_t j(0); j < m_orig_vars.size(); ++j)
          val += m_multipliers[j]
            * ds.finite(convert_index(i), m_orig_vars_indices[j]);
    }
    if (binarized_var == NULL || j != binarized_var_index)
      return val;
    else
      return binary_coloring[val];
  }

  template <typename LA>
  double dataset_view<LA>::vector(size_t i, size_t j) const {
    if (vv_view == VAR_INDICES) {
      if (j >= vector_seq.size()) {
        assert(false);
        return 0;
      }
      return ds.vector(convert_index(i), vv_vector_var_indices[j]);
    } else
      return ds.vector(convert_index(i), j);
  }

  template <typename LA>
  void dataset_view<LA>::convert_assignment(const assignment& orig_r, assignment& new_r) const {
    new_r.finite() = orig_r.finite();
    new_r.vector() = orig_r.vector();
    /*
      foreach(finite_variable* v, keys(orig_r.finite()))
      new_r.finite()[v] = safe_get(orig_r.finite(), v);
      foreach(vector_variable* v, keys(orig_r.vector()))
      new_r.vector()[v] = safe_get(orig_r.vector(), v);
    */
    convert_assignment_(new_r);
  }

  template <typename LA>
  void dataset_view<LA>::convert_record(const record_type& orig_r, record_type& new_r) const {
    if (m_new_var != NULL) {
      new_r.vector() = orig_r.vector();
      tmp_findata = orig_r.finite();
      // Note: tmp_findata is larger than new_r.finite() is.
      std::vector<size_t>& fin = new_r.finite();
      size_t val(0);
      size_t j2(0); // index into m_multipliers
      for (size_t j(0); j < m_orig2new_indices.size(); ++j) {
        if (m_orig2new_indices[j] == std::numeric_limits<size_t>::max()) {
          val += m_multipliers[j2] * tmp_findata[j];
          ++j2;
        } else
          fin[m_orig2new_indices[j]] = tmp_findata[j];
      }
      fin[m_new_var_index] = val;
    } else if (vv_view != VAR_ALL) {
      tmp_findata = orig_r.finite();
      tmp_vecdata = orig_r.vector();
      std::vector<size_t>& fin = new_r.finite();
      vector_type& v = new_r.vector();
      for (size_t j(0); j < vv_finite_var_indices.size(); ++j)
        fin[j] = tmp_findata[vv_finite_var_indices[j]];
      v = tmp_vecdata(vv_vector_var_indices);
      //      for (size_t j(0); j < vv_vector_var_indices.size(); ++j)
      //        v[j] = tmp_vecdata[vv_vector_var_indices[j]];
    } else {
      new_r = orig_r;
    }
    if (binarized_var != NULL) // move this to below if-then
      new_r.finite(binarized_var_index) =
        binary_coloring[new_r.finite(binarized_var_index)];
    new_r.finite_numbering_ptr = finite_numbering_ptr_;
    new_r.vector_numbering_ptr = vector_numbering_ptr_;
  }

  template <typename LA>
  void dataset_view<LA>::revert_merged_value(size_t merged_val,
                                             std::vector<size_t>& orig_vals) const {
    size_t val(merged_val);
    for (size_t j(m_orig_vars_sorted.size()-1); j > 0; --j) {
      orig_vals[j] = val / m_multipliers_sorted[j];
      val = val % m_multipliers_sorted[j];
    }
    orig_vals[0] = val;
  }

  template <typename LA>
  void dataset_view<LA>::revert_merged_value(size_t merged_val,
                                             finite_assignment& orig_vals) const {
    size_t val(merged_val);
    for (size_t j(m_orig_vars_sorted.size()-1); j > 0; --j) {
      orig_vals[m_orig_vars_sorted[j]] = val / m_multipliers_sorted[j];
      val = val % m_multipliers_sorted[j];
    }
    orig_vals[m_orig_vars_sorted[0]] = val;
  }

  template <typename LA>
  boost::shared_ptr<dataset_view<LA> >
  dataset_view<LA>::create_light_view() const {
    vector_dataset<LA>* tmp_ds_ptr =
      new vector_dataset<LA>(ds.datasource_info());
    boost::shared_ptr<dataset_view>
      view_ptr(new dataset_view(tmp_ds_ptr, *this));
    view_ptr->record_view = record_view;
    view_ptr->record_min = record_min;
    view_ptr->record_max = record_max;
    view_ptr->record_indices = record_indices;
    view_ptr->vv_view = vv_view;
    view_ptr->vv_finite_var_indices = vv_finite_var_indices;
    view_ptr->vv_vector_var_indices = vv_vector_var_indices;
    view_ptr->binarized_var = binarized_var;
    view_ptr->binarized_var_index = binarized_var_index;
    view_ptr->binary_var = binary_var;
    view_ptr->binary_coloring = binary_coloring;
    view_ptr->m_new_var = m_new_var;
    view_ptr->m_new_var_index = m_new_var_index;
    view_ptr->m_orig_vars = m_orig_vars;
    view_ptr->m_orig_vars_indices = m_orig_vars_indices;
    view_ptr->m_multipliers = m_multipliers;
    view_ptr->m_orig_vars_sorted = m_orig_vars_sorted;
    view_ptr->m_multipliers_sorted = m_multipliers_sorted;
    view_ptr->m_orig2new_indices = m_orig2new_indices;
    view_ptr->m_new2orig_indices = m_new2orig_indices;
    view_ptr->tmp_findata = tmp_findata;
    view_ptr->tmp_vecdata = tmp_vecdata;
    return view_ptr;
  }

  // Mutating operations: creating views
  //==========================================================================

  template <typename LA>
  void dataset_view<LA>::set_record_range(size_t min, size_t max) {
    assert(min <= max);
    assert(max <= size());
    nrecords = max - min;
    if (record_view == RECORD_RANGE) {
      record_view = RECORD_RANGE;
      record_min = record_min + min;
      record_max = record_min + max;
    } else if (record_view == RECORD_INDICES) {
      record_view = RECORD_INDICES;
      std::vector<size_t> tmp_record_indices;
      for (size_t i = min; i < max; i++)
        tmp_record_indices.push_back(record_indices[i]);
      record_indices = tmp_record_indices;
    } else if (record_view == RECORD_ALL) {
      record_view = RECORD_RANGE;
      record_min = min;
      record_max = max;
    } else
      assert(false);
    if (weighted) {
      vec tmp_weights(max-min);
      for (size_t i = 0; i < max-min; ++i)
        tmp_weights[i] = weights_[min + i];
      weights_ = tmp_weights;
      //        weights_ = weights_.middle(min, max-min);
      // TODO: WHY DOES THE ABOVE LINE NOT WORK?  FIX vector.hpp
    }
  }

  template <typename LA>
  void dataset_view<LA>::set_record_indices(const std::vector<size_t>& indices) {
    if (record_view == RECORD_RANGE) {
      record_indices.clear();
      foreach(size_t i, indices) {
        assert(i <= size());
        record_indices.push_back(record_min + i);
      }
    } else if (record_view == RECORD_INDICES) {
      std::vector<size_t> tmp_record_indices;
      foreach(size_t i, indices) {
        assert(i <= size());
        tmp_record_indices.push_back(record_indices[i]);
      }
      record_indices = tmp_record_indices;
    } else if (record_view == RECORD_ALL) {
      record_indices.clear();
      foreach(size_t i, indices) {
        assert(i <= size());
        record_indices.push_back(i);
      }
    } else
      assert(false);
    nrecords = record_indices.size();
    record_view = RECORD_INDICES;
    if (weighted) {
      ivec tmp_ind(indices.size());
      for (size_t i = 0; i < indices.size(); ++i)
        tmp_ind[i] = indices[i];
      weights_ = weights_(tmp_ind);
    }
  }

  template <typename LA>
  void dataset_view<LA>::set_cross_validation_fold(size_t fold, size_t nfolds,
                                                   bool heldout) {
    assert(fold < nfolds);
    assert((nfolds > 1) && (nfolds <= nrecords));
    size_t lower((size_t)(floor(fold*nrecords / (double)(nfolds))));
    size_t upper((size_t)(floor((fold+1)*nrecords / (double)(nfolds))));
    if (heldout) {
      set_record_range(lower, upper);
    } else {
      std::vector<size_t> indices;
      for (size_t i(0); i < lower; ++i)
        indices.push_back(i);
      for (size_t i(upper); i < nrecords; ++i)
        indices.push_back(i);
      set_record_indices(indices);
    }
  }

  template <typename LA>
  void dataset_view<LA>::save_record_view() {
    assert(saved_record_indices == NULL);
    assert(!weighted);
    saved_record_indices = new std::vector<size_t>();
    switch(record_view) {
    case RECORD_RANGE:
      for (size_t i(record_min); i < record_max; ++i)
        saved_record_indices->push_back(i);
      break;
    case RECORD_INDICES:
      saved_record_indices->operator=(record_indices);
      break;
    case RECORD_ALL:
      for (size_t i(0); i < ds.size(); ++i)
        saved_record_indices->push_back(i);
      break;
    default:
      assert(false);
    }
  }

  template <typename LA>
  void dataset_view<LA>::restore_record_view() {
    assert(saved_record_indices);
    assert(!weighted);
    record_view = RECORD_INDICES;
    record_indices = *saved_record_indices;
    nrecords = record_indices.size();
  }

  template <typename LA>
  void
  dataset_view<LA>::set_variable_indices(const std::set<size_t>& finite_indices,
                                         const std::set<size_t>& vector_indices) {
    if (binarized_var != NULL || m_new_var != NULL) {
      std::cerr << "dataset_view does not support variable views"
                << " simultaneously with binarized and merged variables yet!"
                << std::endl;
      assert(false);
      return;
    }
    // Check indices
    foreach(size_t j, finite_indices) {
      if (j >= num_finite()) {
        std::cerr << "dataset_view::set_variable_indices() was given finite"
                  << " variable index " << j << ", but there are only "
                  << num_finite() << " finite variables." << std::endl;
        assert(false);
      }
    }
    foreach(size_t j, vector_indices) {
      if (j >= num_vector()) {
        std::cerr << "dataset_view::set_variable_indices() was given vector"
                  << " variable index " << j << ", but there are only "
                  << num_vector() << " vector variables." << std::endl;
        assert(false);
      }
    }
    // Construct view
    finite_var_vector new_finite_class_vars;
    vector_var_vector new_vector_class_vars;
    finite_var_vector vv_finite_vars; // Finite variables in view
    vector_var_vector vv_vector_vars; // Vector variables in view
    if (vv_view == VAR_ALL) {
      std::set<finite_variable*>
        old_finite_class_vars(finite_class_vars.begin(),
                              finite_class_vars.end());
      vv_finite_var_indices.clear();
      for (size_t j = 0; j < num_finite(); ++j)
        if (finite_indices.count(j)) {
          vv_finite_vars.push_back(finite_seq[j]);
          vv_finite_var_indices.push_back(j);
          if (old_finite_class_vars.count(finite_seq[j]))
            new_finite_class_vars.push_back(finite_seq[j]);
        }
      std::set<vector_variable*>
        old_vector_class_vars(vector_class_vars.begin(),
                              vector_class_vars.end());
      vv_vector_var_indices.resize(vector_indices.size());
      size_t j2(0); // index into vv_vector_var_indices
      for (size_t j = 0; j < num_vector(); ++j) {
        if (vector_indices.count(j)) {
          vv_vector_vars.push_back(vector_seq[j]);
          vv_vector_var_indices[j2] = j;
          if (old_vector_class_vars.count(vector_seq[j]))
            new_vector_class_vars.push_back(vector_seq[j]);
          ++j2;
        }
      }
    } else {
      // TODO: IMPLEMENT THIS
      assert(false);
    }
    vv_view = VAR_INDICES;
    std::vector<variable::variable_typenames> new_var_type_order;
    size_t j_f = 0, j_v = 0;
    for (size_t j = 0; j < var_type_order.size(); ++j) {
      if (var_type_order[j] == variable::FINITE_VARIABLE) {
        if (finite_indices.count(j_f))
          new_var_type_order.push_back(variable::FINITE_VARIABLE);
        ++j_f;
      } else {
        if (vector_indices.count(j_v))
          new_var_type_order.push_back(variable::VECTOR_VARIABLE);
        ++j_v;
      }
    }
    // Update datasource info
//    finite_vars = finite_domain(vv_finite_vars.begin(), vv_finite_vars.end());
    finite_seq = vv_finite_vars;
    finite_numbering_ptr_->clear();
    dfinite = 0;
    for (size_t j = 0; j < vv_finite_vars.size(); ++j) {
      finite_numbering_ptr_->operator[](vv_finite_vars[j]) = j;
      dfinite += vv_finite_vars[j]->size();
    }
    finite_class_vars = new_finite_class_vars;
//    vector_vars = vector_domain(vv_vector_vars.begin(), vv_vector_vars.end());
    vector_seq = vv_vector_vars;
    vector_numbering_ptr_->clear();
    dvector = 0;
    for (size_t j = 0; j < vv_vector_vars.size(); ++j) {
      vector_numbering_ptr_->operator[](vv_vector_vars[j]) = j;
      dvector += vv_vector_vars[j]->size();
    }
    vector_class_vars = new_vector_class_vars;
    var_type_order = new_var_type_order;
    tmp_findata.resize(ds.num_finite());
    tmp_vecdata.resize(ds.vector_dim());
  }

  template <typename LA>
  void dataset_view<LA>::set_variables(const finite_domain& fvars,
                                       const vector_domain& vvars) {
    std::set<size_t> finite_indices;
    std::set<size_t> vector_indices;
    for (size_t j(0); j < finite_seq.size(); ++j)
      if (fvars.count(finite_seq[j]) != 0)
        finite_indices.insert(j);
    for (size_t j(0); j < vector_seq.size(); ++j)
      if (vvars.count(vector_seq[j]) != 0)
        vector_indices.insert(j);
    set_variable_indices(finite_indices, vector_indices);
  }

  template <typename LA>
  void dataset_view<LA>::set_binary_coloring(finite_variable* original,
                                             finite_variable* binary,
                                             std::vector<size_t> coloring) {
    if (vv_view != VAR_ALL) {
      std::cerr << "dataset_view does not support binarized variables"
                << " simultaneously with variable views yet!" << std::endl;
      assert(false);
      return;
    }
    if (binarized_var != NULL) {
      std::cerr << "dataset_view does not support multiple binarized"
                << " variables yet!" << std::endl;
      assert(false);
      return;
    }
    assert(binary != NULL && binary->size() == 2);
    assert(original != NULL && coloring.size() == original->size());
    assert(has_variable(original));
    for (size_t j = 0; j < coloring.size(); ++j)
      assert(coloring[j] == 0 || coloring[j] == 1);

    binarized_var = original;
    binary_var = binary;
    binary_coloring = coloring;
    binarized_var_index = ds.record_index(original);
//    finite_vars.erase(original);
    for (size_t j = 0; j < finite_seq.size(); j++)
      if (finite_seq[j] == original) {
        finite_seq[j] = binary;
        finite_numbering_ptr_->erase(original);
        finite_numbering_ptr_->operator[](binary) = j;
      }
    for (size_t j = 0; j < finite_class_vars.size(); j++)
      if (finite_class_vars[j] == original)
        finite_class_vars[j] = binary;
    dfinite = dfinite - original->size() + 2;
  }

  template <typename LA>
  void dataset_view<LA>::set_binary_indicator(finite_variable* original,
                                              finite_variable* binary, size_t one_val) {
    assert(original != NULL && one_val < original->size());
    std::vector<size_t> coloring(original->size(), 0);
    coloring[one_val] = 1;
    set_binary_coloring(original, binary, coloring);
  }

  template <typename LA>
  void dataset_view<LA>::set_merged_variables(finite_var_vector original_vars,
                                              finite_variable* new_var) {
    // Check input.
    if (m_new_var != NULL) {
      std::cerr << "dataset_view::set_merged_variables() may not be called"
                << " on a view which already has merged variables."
                << std::endl;
      assert(false);
      return;
    }
    if (vv_view != VAR_ALL) {
      std::cerr << "dataset_view does not support merged variables"
                << " simultaneously with variable views yet!" << std::endl;
      assert(false);
      return;
    }
    assert(new_var && !has_variable(new_var));
    assert(original_vars.size() > 0);
    size_t new_size(1);
    std::set<finite_variable*> tmp_fin_class_vars(finite_class_vars.begin(),
                                                  finite_class_vars.end());
    bool is_class = tmp_fin_class_vars.count(original_vars[0]);
    for (size_t j(0); j < original_vars.size(); ++j) {
      assert(original_vars[j] != NULL &&
             has_variable(original_vars[j]) &&
             original_vars[j] != binarized_var);
      new_size *= original_vars[j]->size();
      if (is_class) {
        if (!tmp_fin_class_vars.count(original_vars[j])) {
          assert(false);
          return;
        }
      } else {
        if (tmp_fin_class_vars.count(original_vars[j])) {
          assert(false);
          return;
        }
      }
    }
    if (new_var->size() != new_size) {
      std::cerr << "dataset_view::set_merged_variables() was given a new "
                << "finite variable of size " << new_var->size()
                << " but should have received one of size " << new_size
                << std::endl;
      assert(false);
      return;
    }
    // Construct the new finite variable ordering, putting the new variable
    //  at the end of the ordering.
    m_new_var = new_var;
    m_new_var_index = num_finite() - original_vars.size();
    m_orig_vars_sorted = original_vars;
    m_orig_vars_indices.clear();
    m_orig2new_indices.clear();
    m_orig2new_indices.resize(num_finite(), 0);
    m_new2orig_indices.resize(num_finite() - original_vars.size() + 1);
    m_multipliers_sorted.resize(original_vars.size());
    tmp_findata.resize(num_finite());
    for (size_t j(0); j < original_vars.size(); ++j) {
      m_orig2new_indices[ds.record_index(original_vars[j])]
        = std::numeric_limits<size_t>::max();
      if (j == 0)
        m_multipliers_sorted[0] = 1;
      else
        m_multipliers_sorted[j] = original_vars[j-1]->size() * m_multipliers_sorted[j-1];
      m_orig_vars_indices.push_back(ds.record_index(original_vars[j]));
    }
    size_t j2(0); // index in new findata corresponding to j
    for (size_t j(0); j < num_finite(); ++j) {
      if (m_orig2new_indices[j] != std::numeric_limits<size_t>::max()) {
        m_orig2new_indices[j] = j2;
        m_new2orig_indices[j2] = j;
        ++j2;
      }
    }
    m_new2orig_indices.back() = std::numeric_limits<size_t>::max();
    assert(m_orig_vars_sorted.size() == original_vars.size()); // check uniqueness
    // Fix indices for binarizing variables.
    if (binarized_var != NULL)
      binarized_var_index = m_orig2new_indices[binarized_var_index];
    // Fix datasource finite variable and variable ordering info.
    for (size_t j(0); j < original_vars.size(); ++j) {
//      finite_vars.erase(original_vars[j]);
      dfinite -= original_vars[j]->size();
    }
//    finite_vars.insert(new_var);
    dfinite += new_var->size();
    if (is_class) {
      finite_class_vars.clear();
      for (size_t j(0); j < original_vars.size(); ++j)
        tmp_fin_class_vars.erase(original_vars[j]);
      foreach(finite_variable* f, tmp_fin_class_vars)
        finite_class_vars.push_back(f);
      finite_class_vars.push_back(new_var);
    }
    finite_var_vector tmp_finite_seq;
    for (size_t j(0); j < m_new2orig_indices.size() - 1; ++j) {
      tmp_finite_seq.push_back(finite_seq[m_new2orig_indices[j]]);
    }
    tmp_finite_seq.push_back(new_var);
    std::vector<variable::variable_typenames> tmp_var_type_order;
    j2 = 0; // index into original finite_seq
    std::set<size_t> tmp_orig_vars_indices_set(m_orig_vars_indices.begin(),
                                               m_orig_vars_indices.end());
    for (size_t j(0); j < var_type_order.size(); ++j) {
      if (var_type_order[j] == variable::FINITE_VARIABLE) {
        if (!tmp_orig_vars_indices_set.count(j2)) // if not in merged vars
          tmp_var_type_order.push_back(variable::FINITE_VARIABLE);
        ++j2;
      } else {
        tmp_var_type_order.push_back(variable::VECTOR_VARIABLE);
      }
    }
    tmp_var_type_order.push_back(variable::FINITE_VARIABLE);
    finite_seq = tmp_finite_seq;
    size_t nfinite(0);
    foreach(finite_variable* v, finite_seq)
      finite_numbering_ptr_->operator[](v) = nfinite++;
    var_type_order = tmp_var_type_order;
    // Reorder m_orig_vars, m_orig_vars_indices, m_multipliers so that they
    //  follow the order of the original variables in the original dataset.
    // m_orig_vars_indices-->index in m_orig_vars
    std::map<size_t, size_t> tmp_orderstats;
    m_orig_vars.clear();
    std::vector<size_t> tmp_m_orig_vars_indices;
    m_multipliers.clear();
    for (size_t j(0); j < m_orig_vars_sorted.size(); ++j)
      tmp_orderstats[m_orig_vars_indices[j]] = j;
    for (size_t j(0); j < ds.num_finite(); ++j)
      if (tmp_orig_vars_indices_set.count(j)) { // if in merged vars
        m_orig_vars.push_back(m_orig_vars_sorted[tmp_orderstats[j]]);
        tmp_m_orig_vars_indices.push_back(j);
        m_multipliers.push_back(m_multipliers_sorted[tmp_orderstats[j]]);
      }
    m_orig_vars_indices = tmp_m_orig_vars_indices;
  } // set_merged_variables()

  template <typename LA>
  void dataset_view<LA>::restrict_to_assignment(const finite_assignment& fa) {
    std::vector<size_t> offsets(fa.size(), 0); // indices of vars in records
    std::vector<size_t> vals(fa.size(), 0);    // corresponding values
    size_t i(0);
    for (finite_assignment::const_iterator it(fa.begin());
         it != fa.end(); ++it) {
      offsets[i] = this->record_index(it->first);
      vals[i] = it->second;
      ++i;
    }
    std::vector<size_t> indices;
    i = 0;
    foreach(const record_type& r, this->records()) {
      bool fits(true);
      for (size_t j(0); j < offsets.size(); ++j) {
        if (r.finite(offsets[j]) != vals[j]) {
          fits = false;
          break;
        }
      }
      if (fits)
        indices.push_back(i);
      ++i;
    }
    set_record_indices(indices);
  } // restrict_to_assignment()

  // Mutating operations: datasets
  //==========================================================================

  template <typename LA>
  void dataset_view<LA>::randomize(double random_seed) {
    boost::mt11213b rng(static_cast<unsigned>(random_seed));
    if (record_view == RECORD_RANGE) {
      record_indices.clear();
      for (size_t i = record_min; i < record_max; ++i)
        record_indices.push_back(i);
    } else if (record_view == RECORD_INDICES) {
    } else if (record_view == RECORD_ALL) {
      record_indices.clear();
      for (size_t i = 0; i < nrecords; ++i)
        record_indices.push_back(i);
    } else
      assert(false);
    for (size_t i = 0; i < nrecords-1; ++i) {
      size_t j((size_t)(boost::uniform_int<int>(i,nrecords-1)(rng)));
      size_t tmp(record_indices[i]);
      record_indices[i] = record_indices[j];
      record_indices[j] = tmp;
      if (weighted) {
        double tmpw(weights_[i]);
        weights_[i] = weights_[j];
        weights_[j] = tmpw;
      }
    }
    record_view = RECORD_INDICES;
  }

  // Save and load methods
  //==========================================================================

  template <typename LA>
  void dataset_view<LA>::save(std::ofstream& out) const {
    out << record_view << " ";
    if (record_view == RECORD_RANGE)
      out << record_min << " " << record_max << " ";
    else if (record_view == RECORD_INDICES)
      out << record_indices << " ";
    out << (binarized_var == NULL ? "0 " : "1 ");
    if (binarized_var != NULL)
      out << binarized_var_index << " " << binary_coloring << " ";
    out << (m_new_var == NULL ? "0 " : "1 ");
    if (m_new_var != NULL)
      out << m_orig_vars_indices << " "; // TODO: FIX THIS!
    out  << "\n";
    if (vv_view != VAR_ALL)
      assert(false);
  }

  template <typename LA>
  void dataset_view<LA>::save(const std::string& filename) const {
    std::ofstream out(filename.c_str(), std::ios::out);
    save(out);
    out.flush();
    out.close();
  }

  template <typename LA>
  bool dataset_view<LA>::load(std::ifstream& in, finite_variable* binary_var_,
                              finite_variable* m_new_var_) {
    std::string line;
    getline(in, line);
    std::istringstream is(line);
    size_t tmpsize;
    // Record views
    if (!(is >> tmpsize))
      assert(false);
    record_view = static_cast<record_view_type>(tmpsize);
    if (record_view == RECORD_RANGE) {
      if (!(is >> record_min))
        assert(false);
      if (!(is >> record_max))
        assert(false);
    } else if (record_view == RECORD_INDICES)
      read_vec(is, record_indices);
    // Variable views
    vv_view = VAR_ALL; // TODO
    // Binarized views
    if (!(is >> tmpsize))
      assert(false);
    if (tmpsize == 1) {
      if (!(is >> binarized_var_index))
        assert(false);
      read_vec(is, binary_coloring);
      assert(binarized_var_index < ds.num_finite());
      set_binary_coloring(ds.finite_list()[binarized_var_index],
                          binary_var_, binary_coloring);
    }
    // Merged finite variable views  // TODO: FIX THIS!
    if (!(is >> tmpsize))
      assert(false);
    if (tmpsize == 1) {
      read_vec(is, m_orig_vars_indices);
      finite_var_vector orig_vars;
      foreach(size_t j, m_orig_vars_indices) {
        assert(j < ds.num_finite());
        orig_vars.push_back(ds.finite_list()[j]);
      }
      set_merged_variables(orig_vars, m_new_var_);
    }
    return true;
  }

  template <typename LA>
  bool dataset_view<LA>::load(const std::string& filename, finite_variable* binary_var_,
                              finite_variable* m_new_var_) {
    std::ifstream in(filename.c_str(), std::ios::in);
    bool val = load(in, binary_var_, m_new_var_);
    in.close();
    return val;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#include <sill/learning/dataset/vector_dataset.hpp>

#endif
