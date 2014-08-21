#ifndef SILL_DATASOURCE_HPP
#define SILL_DATASOURCE_HPP

#include <map>

#include <sill/base/assignment.hpp>
#include <sill/base/variable_type_group.hpp>
#include <sill/copy_ptr.hpp>
#include <sill/iterator/map_key_iterator.hpp>
#include <sill/learning/dataset/datasource_info_type.hpp>
#include <sill/learning/dataset/record_conversions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Datasource.  Base class for datasets and data oracles.
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   */
  class datasource {

    // Public types
    //==========================================================================
  public:

    typedef map_key_iterator<std::map<finite_variable*, size_t> >
    finite_var_iterator;

    typedef map_key_iterator<std::map<vector_variable*, size_t> >
    vector_var_iterator;

    // Constructors
    //==========================================================================

    //! Empty datasource
    datasource();

    //! Constructs the datasource with the given sequence of variables.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    datasource(const finite_var_vector& finite_vars,
               const vector_var_vector& vector_vars,
               const std::vector<variable::variable_typenames>& var_type_order);

    //! Constructs the datasource with the given sequence of variables.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    datasource(const forward_range<finite_variable*>& finite_vars,
               const forward_range<vector_variable*>& vector_vars,
               const std::vector<variable::variable_typenames>& var_type_order);

    //! Constructs the datasource with the given sequence of variables.
    //! @param var_info    info from calling datasource_info()
    explicit datasource(const datasource_info_type& info);

    virtual ~datasource() { }

    //! This method is like a constructor but is virtualized.
    //! @param info    info from calling datasource_info()
    virtual void reset(const datasource_info_type& info);

    void save(oarchive& a) const;

    void load(iarchive& a);

    // Variables
    //==========================================================================

    //! Returns the variable set of this dataset
    domain variables() const;

    //! Returns the finite variables of this dataset
    std::pair<finite_var_iterator,finite_var_iterator>
    finite_variables() const;
//    const finite_domain& finite_variables() const;

    //! Returns the vector variables of this datatset
    std::pair<vector_var_iterator,vector_var_iterator>
    vector_variables() const;
//    const vector_domain& vector_variables() const;

    //! Constructs and returns the list of variables for this dataset.
    var_vector variable_list() const;

    //! Returns the finite variables in the natural order
    const finite_var_vector& finite_list() const;

    //! Returns the vector variables in the natural order
    const vector_var_vector& vector_list() const;

    //! Constructs and returns a list of variables of the specified type,
    //! in the natural order.
    template <typename VarType>
    typename variable_type_group<VarType>::var_vector_type
    variable_sequence() const;

    //! Returns the finite class variable.
    //! NOTE: This asserts false if there is not exactly 1 finite class var.
    finite_variable* finite_class_variable() const;

    //! Returns the vector class variable (or NULL if none exists).
    //! NOTE: This asserts false if there is not exactly 1 vector class var.
    vector_variable* vector_class_variable() const;

    //! Returns the finite class variables (if any)
    const finite_var_vector& finite_class_variables() const;

    //! Returns the vector class variables (if any)
    const vector_var_vector& vector_class_variables() const;

    //! Indicates if this datasource has the given variable.
    bool has_variable(finite_variable* v) const;

    //! Indicates if this datasource has the given variable.
    bool has_variable(vector_variable* v) const;

    //! Indicates if this datasource has the given variable.
    bool has_variable(variable* v) const;

    //! Indicates if this datasource has the given variables.
    bool has_variables(const finite_domain& vars) const;

    //! Indicates if this datasource has the given variables.
    bool has_variables(const vector_domain& vars) const;

    //! Indicates if this datasource has the given variables.
    bool has_variables(const domain& vars) const;

    // Dimensionality of variables
    //==========================================================================

    //! Returns the total number of variables
    size_t num_variables() const;

    //! Returns the number of finite variables
    size_t num_finite() const;

    //! Returns the number of vector variables
    size_t num_vector() const;

    //! Returns the total dimensionality of finite variables,
    //! i.e., the sum of their sizes
    size_t finite_dim() const;

    //! Returns the total dimensionality of vector variables,
    //! i.e., the sum of their sizes
    size_t vector_dim() const;

    // Indexing of variables
    //==========================================================================

    //! Order of variable types in dataset's natural order
    const std::vector<variable::variable_typenames>&
    variable_type_order() const;

    //! The variables in this dataset, in the order used in the dataset file
    //!  (or other source).
    var_vector var_order() const;

    //! Returns the index of the given variable in the order returned by
    //! var_order().
//    size_t var_order_index(variable* v) const;

    //! Computes a mapping from variables to their indices in dataset's
    //! natural order.
    //! NOTE: This constructs the index!
    std::map<variable*, size_t> variable_order_map() const;

    /*
    //! Returns the index of a finite variable in the order of finite_list().
    size_t variable_index(finite_variable* v) const;

    //! Returns the index of a vector variable in the order of vector_list().
    size_t variable_index(vector_variable* v) const;
    */

    //! Returns the index of the given finite variable in records.
    //! (This is the same as in the natural order.)
    size_t record_index(finite_variable* v) const;

    //! Returns the first index of the given vector variable in records.
    //! (This is the same as the natural order except that vector variables
    //!  with length > 1 take up multiple indices.)
    size_t record_index(vector_variable* v) const;

    //! Returns the indices (in records) of the given vector variables.
    //! @todo Check to make sure this works with variables of length > 1.
    uvec vector_indices(const vector_domain& vars) const;

    //! Returns the indices (in records) of the given vector variables.
    //! @todo Check to make sure this works with variables of length > 1.
    uvec vector_indices(const vector_var_vector& vars) const;

    //! Mapping from finite variables to indices in finite component of record.
    const std::map<finite_variable*, size_t>& finite_numbering() const;

    //! Mapping from finite variables to indices in finite component of record.
    copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr() const;

    //! Mapping from vector variables to indices in vector component of record.
    const std::map<vector_variable*, size_t>& vector_numbering() const;

    //! Mapping from vector variables to indices in vector component of record.
    copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr() const;

    // General info and helpers
    //==========================================================================

    //! Returns all variable type and order information.
    datasource_info_type datasource_info() const;

    //! Compare this datasource with another to see if they contain comparable
    //! data.  Return true if yes, else false.
    bool comparable(const datasource& ds) const;

    //! Print info about datasource to STDERR for debugging.
    void print_datasource_info() const;

    // Setters
    //==========================================================================

    //! Sets the finite class variables to a single variable
    //! If no arguments, then set to none.
    void set_finite_class_variable(finite_variable* class_var = NULL);
    
    //! Sets the finite class variables
    //! @todo This should check to make sure the variables are unique.
    void set_finite_class_variables(const finite_var_vector& class_vars);

    //! Sets the finite class variables
    void set_finite_class_variables(const finite_domain& class_vars);

    //! Sets the vector class variables to a single variable
    //! If no arguments, then set to none.
    void set_vector_class_variable(vector_variable* class_var = NULL);

    //! Sets the vector class variables
    //! @todo This should check to make sure the variables are unique.
    void set_vector_class_variables(const vector_var_vector& class_vars);

    //! Sets the vector class variables
    void set_vector_class_variables(const vector_domain& class_vars);

    // Protected data members
    //==========================================================================
  protected:

    //! Type of map value from a finite variable to its index
    typedef std::pair<finite_variable*, size_t> finite_var_index_pair;

    //! Type of map value from a vector variable to the first index in the
    //! space allocated for it
    typedef std::pair<vector_variable*, size_t> vector_var_index_pair;

    //! The finite variables in this dataset
//    finite_domain finite_vars;

    //! The finite variables in this dataset,
    //!  in the order used in the dataset file (or other source).
    //!  (This is only valid if var_type_order is specified.)
    finite_var_vector finite_seq;

    //! Mapping from finite variable identifiers to the column indices.
    //! This is kept in a copy pointer since records store the same info;
    //! this allows records to remain valid after the dataset has been deleted.
    copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr_;

    //! Total dimensionality of the finite variables (when each variable of
    //! arity N is turned into N indicator variables); i.e., the sum of the
    //! variable sizes.
    size_t dfinite;

    //! Finite class variables (if any)
    finite_var_vector finite_class_vars;

    //! The vector variables in this dataset
//    vector_domain vector_vars;

    //! The vector variables in this dataset,
    //!  in the order used in the dataset file (or other source).
    //!  (This is only valid if var_type_order is specified.)
    vector_var_vector vector_seq;

    //! Mapping from vector variable identifiers to the first indices of the
    //! column ranges.
    //! This is kept in a copy pointer since records store the same info;
    //! this allows records to remain valid after the dataset has been deleted.
    copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr_;

    //! Total dimensionality of the vector variables (when concatenated);
    //! i.e., the sum of the variable sizes.
    size_t dvector;

    //! Vector class variables (if any)
    vector_var_vector vector_class_vars;

    //! Order of variable types in dataset's natural order
    std::vector<variable::variable_typenames> var_type_order;

    //! Mapping from variables to their indices in dataset's natural order.
//    std::map<variable*, size_t> var_order_map;

    //! Mapping from vector variables to their indices in vector_seq.
//    std::map<vector_variable*, size_t> vector_var_order_map;

    // Protected helper functions
    //==========================================================================

    //! Initializes the index maps to the variables in the given order.
    void initialize();

    //! Converts the given finite record data into a finite assignment.
    void convert_finite_record2assignment(const std::vector<size_t>& findata,
                                          finite_assignment& fa) const;

    //! Converts the given vector record data into a vector assignment.
    void convert_vector_record2assignment(const vec& vecdata,
                                          vector_assignment& va) const;

    //! Converts the given vector record data into a vector assignment.
    template <typename T, typename Index>
    void convert_vector_record2assignment(const sparse_vector<T,Index>& vecdata,
                                          vector_assignment& va) const {
      assert(vecdata.size() == dvector);
      va.clear();
      foreach(const vector_var_index_pair& p, *vector_numbering_ptr_) {
        vec tmpvec(p.first->size());
        for(size_t j = 0; j < p.first->size(); j++)
          tmpvec[j] = vecdata[j+p.second];
        va[p.first] = tmpvec;
      }
    }

    //! Converts the given finite assignment into finite record data.
    void convert_finite_assignment2record(const finite_assignment& fa,
                                          std::vector<size_t>& findata) const;

    //! Converts the given vector assignment into vector record data.
    template <typename VecType>
    void convert_vector_assignment2record
    (const vector_assignment& va, VecType& vecdata) const {
      vector_assignment2vector(va, vector_seq, vecdata);
    }

    //! Add a finite variable to this datasource.
    //! This adds the variable to the end of the variable ordering.
    void add_finite_variable(finite_variable* v, bool make_class = false);

    //! Add a vector variable to this datasource.
    //! This adds the variable to the end of the variable ordering.
    void add_vector_variable(vector_variable* v, bool make_class = false);

  }; // class datasource

  // Specializations of templated functions in datasource
  // ===========================================================================

  //! Specialization for variable.
  template <>
  var_vector datasource::variable_sequence<variable>() const;

  //! Specialization for finite_variable.
  template <>
  finite_var_vector datasource::variable_sequence<finite_variable>() const;

  //! Specialization for vector_variable.
  template <>
  vector_var_vector datasource::variable_sequence<vector_variable>() const;

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DATASOURCE_HPP
