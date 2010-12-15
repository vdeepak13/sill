
#ifndef PRL_DATASET_VIEW_HPP
#define PRL_DATASET_VIEW_HPP

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>

#include <boost/iterator.hpp>
#include <boost/multi_array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <prl/base/assignment.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/range/concepts.hpp>

#include <prl/range/algorithm.hpp>

#include <prl/macros_def.hpp>

namespace prl {

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
   */
  class dataset_view : public dataset {

    // Protected data members
    //==========================================================================
  protected:

    //! Used for deleting the dataset to make a light class which can convert
    //! records according to the view.
    vector_dataset* ds_ptr;

    //! Dataset
    const dataset& ds;

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
    mutable vec tmp_vecdata;

    // Protected functions
    //==========================================================================

    //! Translates index i (into view) into index for actual dataset.
    size_t convert_index(size_t i) const;

    //! Constructor for a view without associated data;
    //! used by create_light_view().
    dataset_view(vector_dataset* ds_ptr, const dataset_view& ds_view_source)
      : dataset(ds_view_source), ds_ptr(ds_ptr), ds(*ds_ptr),
        record_view(RECORD_ALL), saved_record_indices(NULL),
        vv_view(VAR_ALL), binarized_var(NULL), m_new_var(NULL) { }

    // Protected functions required by record
    //==========================================================================

    //! Used by load_assignment() and convert_assignment()
    void convert_assignment_(assignment& a) const;

    //! Load datapoint i into assignment a
    void load_assignment(size_t i, prl::assignment& a) const;

    //! Load record i into r
    void load_record(size_t i, record& r) const;

    //! Load finite data for datapoint i into findata
    void load_finite(size_t i, std::vector<size_t>& findata) const;

    //! Load vector data for datapoint i into vecdata
    void load_vector(size_t i, vec& vecdata) const;

    //! ONLY for datasets which use assignments as native types:
    //!  Load the pointer to datapoint i into (*a).
    void load_assignment_pointer(size_t i, assignment** a) const;

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
      : dataset(ds_view), ds_ptr(ds_view.ds_ptr), ds(ds_view.ds),
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
    explicit dataset_view(const dataset& ds, bool keep_weights = false)
      : dataset(ds), ds_ptr(NULL), ds(ds),
        record_view(RECORD_ALL), saved_record_indices(NULL),
        vv_view(VAR_ALL), binarized_var(NULL), m_new_var(NULL) {
      if (keep_weights && ds.weighted) {
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

    //! Return capacity
    size_t capacity() const { return ds.capacity(); }

    //! Element access: record i, finite variable j (in the order finite_list())
    //! Note: Full record retrievals are more efficient than this function.
    size_t finite(size_t i, size_t j) const;

    using dataset::vector;

    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! Note: Full record retrievals are more efficient than this function.
    double vector(size_t i, size_t j) const;

    //! Convert an assignment from the original dataset into one for this view.
    //! Note: The caller must know what the original dataset looked like!
    void convert_assignment(const assignment& orig_r, assignment& new_r) const;

    //! Convert a record from the original dataset into one for this view.
    //! Note: The caller must know what the original dataset looked like!
    //! @param new_r  Pre-allocated and properly sized record.
    void convert_record(const record& orig_r, record& new_r) const;

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
    void insert(const record& r, double w = 1) {
      assert(false);
    }

    //! Adds a new record with finite variable values fvals and vector variable
    //! values vvals, with weight w (default = 1).
    void insert(const std::vector<size_t>& fvals, const vec& vvals,
                double w = 1) {
      assert(false);
    }

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const assignment& a, double w = 1) {
      assert(false);
    }

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const std::vector<size_t>& fvals,
                    const vec& vvals, double w = 1) {
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

  };  // class dataset_view

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
