
#ifndef PRL_DATASOURCE_HPP
#define PRL_DATASOURCE_HPP

#include <map>

#include <prl/base/assignment.hpp>
#include <prl/copy_ptr.hpp>
#include <prl/learning/dataset/datasource_info_type.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Datasource.
   * \author Joseph Bradley
   * \ingroup learning_dataset
   */
  class datasource {

    // Protected data members
    //==========================================================================
  protected:

    //! Type of map value from a finite variable to its index
    typedef std::pair<finite_variable*, size_t> finite_var_index_pair;

    //! Type of map value from a vector variable to the first index in the
    //! space allocated for it
    typedef std::pair<vector_variable*, size_t> vector_var_index_pair;

    //! The finite variables in this dataset
    finite_domain finite_vars;

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
    vector_domain vector_vars;

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
    std::map<variable*, size_t> var_order_map;

    //! Mapping from vector variables to their indices in vector_seq.
    std::map<vector_variable*, size_t> vector_var_order_map;

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

    //! Converts the given finite assignment into finite record data.
    void convert_finite_assignment2record(const finite_assignment& fa,
                                          std::vector<size_t>& findata) const;

    //! Converts the given vector assignment into vector record data.
    void convert_vector_assignment2record(const vector_assignment& va,
                                          vec& vecdata) const;

    //! Add a finite variable to this datasource.
    //! This adds the variable to the end of the variable ordering.
    void add_finite_variable(finite_variable* v, bool make_class = false);

    //! Add a vector variable to this datasource.
    //! This adds the variable to the end of the variable ordering.
    void add_vector_variable(vector_variable* v, bool make_class = false);

    // Constructors
    //==========================================================================
  public:

    //! Empty datasource
    datasource()
      : finite_numbering_ptr_(new std::map<finite_variable*, size_t>()),
        dfinite(0),
        vector_numbering_ptr_(new std::map<vector_variable*, size_t>()),
        dvector(0) { }

    //! Constructs the datasource with the given sequence of variables.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    datasource(const finite_var_vector& finite_vars,
               const vector_var_vector& vector_vars,
               const std::vector<variable::variable_typenames>& var_type_order)
      : finite_vars(finite_vars.begin(), finite_vars.end()), 
        finite_seq(finite_vars),
        finite_numbering_ptr_(new std::map<finite_variable*, size_t>()),
        vector_vars(vector_vars.begin(), vector_vars.end()), 
        vector_seq(vector_vars),
        vector_numbering_ptr_(new std::map<vector_variable*, size_t>()),
        var_type_order(var_type_order) {
      initialize();
    }

    //! Constructs the datasource with the given sequence of variables.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    datasource(const forward_range<finite_variable*>& finite_vars,
               const forward_range<vector_variable*>& vector_vars,
               const std::vector<variable::variable_typenames>& var_type_order)
      : finite_vars(boost::begin(finite_vars), boost::end(finite_vars)),
        finite_seq(boost::begin(finite_vars), boost::end(finite_vars)),
        finite_numbering_ptr_(new std::map<finite_variable*, size_t>()),
        vector_vars(boost::begin(vector_vars), boost::end(vector_vars)),
        vector_seq(boost::begin(vector_vars), boost::end(vector_vars)),
        vector_numbering_ptr_(new std::map<vector_variable*, size_t>()),
        var_type_order(var_type_order) {
      initialize();
    }

    //! Constructs the datasource with the given sequence of variables.
    //! @param var_info    info from calling datasource_info()
    explicit datasource(const datasource_info_type& info)
      : finite_vars(info.finite_seq.begin(), info.finite_seq.end()), 
        finite_seq(info.finite_seq),
        finite_numbering_ptr_(new std::map<finite_variable*, size_t>()),
        finite_class_vars(info.finite_class_vars),
        vector_vars(info.vector_seq.begin(), info.vector_seq.end()), 
        vector_seq(info.vector_seq),
        vector_numbering_ptr_(new std::map<vector_variable*, size_t>()),
        vector_class_vars(info.vector_class_vars),
        var_type_order(info.var_type_order) {
      initialize();
    }

    virtual ~datasource() { }

    // Variables
    //==========================================================================

    //! Returns the variable set of this dataset
    domain variables() const {
      domain d1(finite_vars.begin(), finite_vars.end());
      domain d2(vector_vars.begin(), vector_vars.end());
      return set_union(d1, d2);
    }

    //! Returns the finite variables of this dataset
    const finite_domain& finite_variables() const {
      return finite_vars;
    }

    //! Returns the vector variables of this datatset
    const vector_domain& vector_variables() const {
      return vector_vars;
    }

    //! Returns the finite variables in the natural order
    const finite_var_vector& finite_list() const {
      return finite_seq;
    }

    //! Returns the vector variables in the natural order
    const vector_var_vector& vector_list() const {
      return vector_seq;
    }

    //! Returns the finite class variables (if any)
    const finite_var_vector& finite_class_variables() const {
      return finite_class_vars;
    }

    //! Returns the vector class variables (if any)
    const vector_var_vector& vector_class_variables() const {
      return vector_class_vars;
    }

    // Dimensionality of variables
    //==========================================================================

    //! Returns the total number of variables
    size_t num_variables() const {
      return vector_vars.size() + finite_vars.size();
    }

    //! Returns the number of finite variables
    size_t num_finite() const {
      return finite_vars.size();
    }

    //! Returns the number of vector variables
    size_t num_vector() const {
      return vector_vars.size();
    }

    //! Returns the total dimensionality of finite variables,
    //! i.e., the sum of their sizes
    size_t finite_dim() const {
      return dfinite;
    }

    //! Returns the total dimensionality of vector variables,
    //! i.e., the sum of their sizes
    size_t vector_dim() const {
      return dvector;
    }

    // Indexing of variables
    //==========================================================================

    //! Order of variable types in dataset's natural order
    const std::vector<variable::variable_typenames>&
    variable_type_order() const {
      return var_type_order;
    }

    //! The variables in this dataset, in the order used in the dataset file
    //!  (or other source).
    var_vector var_order() const;

    //! Returns the index of the given variable in the order returned by
    //! var_order().
    size_t var_order_index(variable* v) const;

    //! Returns the index of a finite variable in the order of finite_list().
    size_t variable_index(finite_variable* v) const {
      return record_index(v);
    }

    //! Returns the index of a vector variable in the order of vector_list().
    size_t variable_index(vector_variable* v) const {
      return safe_get(vector_var_order_map, v);
    }

    //! Returns the index of the given finite variable in records.
    //! (This is the same as in the natural order.)
    size_t record_index(finite_variable* v) const {
      return safe_get(*finite_numbering_ptr_, v);
    }

    //! Returns the first index of the given vector variable in records.
    //! (This is the same as the natural order except that vector variables
    //!  with length > 1 take up multiple indices.)
    size_t record_index(vector_variable* v) const {
      return safe_get(*vector_numbering_ptr_, v);
    }

    //! Returns the indices (in records) of the given vector variables.
    //! @todo Check to make sure this works with variables of length > 1.
    ivec vector_indices(const vector_domain& vars) const;

    //! Returns the indices (in records) of the given vector variables.
    //! @todo Check to make sure this works with variables of length > 1.
    ivec vector_indices(const vector_var_vector& vars) const;

    //! Mapping from finite variables to indices in finite component of record.
    const std::map<finite_variable*, size_t>& finite_numbering() const {
      return *finite_numbering_ptr_;
    }

    //! Mapping from finite variables to indices in finite component of record.
    copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr() const {
      return finite_numbering_ptr_;
    }

    //! Mapping from vector variables to indices in vector component of record.
    const std::map<vector_variable*, size_t>& vector_numbering() const {
      return *vector_numbering_ptr_;
    }

    //! Mapping from vector variables to indices in vector component of record.
    copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr() const {
      return vector_numbering_ptr_;
    }

    // General info and helpers
    //==========================================================================

    //! Returns a tuple with all variable type and order information.
    datasource_info_type datasource_info() const {
      return datasource_info_type(finite_seq, vector_seq, var_type_order,
                                  finite_class_vars, vector_class_vars);
    }

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

  }; // class datasource

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_DATASOURCE_HPP
