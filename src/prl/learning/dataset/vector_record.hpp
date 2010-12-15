
#ifndef PRL_VECTOR_RECORD_HPP
#define PRL_VECTOR_RECORD_HPP

#include <map>

#include <prl/base/vector_assignment.hpp>
#include <prl/base/stl_util.hpp>
#include <prl/copy_ptr.hpp>
#include <prl/math/vector.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * A type that provides a mutable view of a single datapoint's vector data.
   * This provides flexible access to data in 2 forms:
   *  - vector form: a vector of values (for algorithms which know the
   *                 ordering of variables)
   *  - assignment form: a mapping from variables to values
   *
   * Records can store their own data but can also hold pointers into data
   * stored in other structures.
   * When a record is copied, the copy will store its own data.
   *
   * \author Joseph Bradley, Stanislav Funiak
   * \todo Allow vector_records to be restricted to a set of variables.
   * @todo Change assignment(), finite_assignment(), etc. to allow efficient
   *       access for datasets whose native type is assignment.
   * \ingroup learning_dataset
   */
  class vector_record {

    // Public types and data members
    //==========================================================================
  public:

    //! Type of variable used.
    typedef vector_variable variable_type;

    //! Type of map value from a vector variable to the first index in a
    //! vector of values.
    typedef std::pair<vector_variable*, size_t> vector_var_index_pair;

    //! Map from vector variables to the first indices for those variables'
    //! values in a vector of values.
    copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr;

    //! True if record owns its data
    bool vec_own;

    //! Handle for vector variable values for record
    vec* vec_ptr;

    /*
// TODO: FINISH RECORD VIEWS
    //! Indicates if this is a view.
    bool is_vec_view;
    //! Indices of vector variable components (elements of vec_ptr) in view.
    std::vector<size_t> vec_view;
    */

    // Constructors
    //==========================================================================

    //! Constructs an empty vector_record which owns its data.
    vector_record()
      : vector_numbering_ptr(new std::map<vector_variable*, size_t>()),
        vec_own(true), vec_ptr(new vec()) {
    }

    //! Constructor for a vector_record which owns its data.
    //! @param vector_dim  Size of the vector data.
    vector_record
    (copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
     size_t vector_dim)
      : vector_numbering_ptr(vector_numbering_ptr), vec_own(true),
        vec_ptr(new vec(vector_dim)) {
    }

    //! Constructor for a vector_record which uses data from its creator
    vector_record
    (copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
     vec* vec_ptr)
      : vector_numbering_ptr(vector_numbering_ptr),
        vec_own(false), vec_ptr(vec_ptr) {
    }

    //! Constructor for a vector_record which owns its data.
    explicit vector_record(const vector_var_vector& vector_seq,
                           size_t vector_dim)
      : vector_numbering_ptr(new std::map<vector_variable*,size_t>()),
        vec_own(true), vec_ptr(new vec(vector_dim)) {
      size_t k(0);
      for (size_t j(0); j < vector_seq.size(); ++j) {
        vector_numbering_ptr->operator[](vector_seq[j]) = k;
        k += vector_seq[j]->size();
      }
      assert(k == vector_dim);
    }

    //! Constructor for a vector_record which owns its data.
    explicit vector_record(const vector_var_vector& vector_seq)
      : vector_numbering_ptr(new std::map<vector_variable*,size_t>()),
        vec_own(true), vec_ptr(NULL) {
      size_t k(0);
      for (size_t j(0); j < vector_seq.size(); ++j) {
        vector_numbering_ptr->operator[](vector_seq[j]) = k;
        k += vector_seq[j]->size();
      }
      vec_ptr = new vec(k);
    }

    //! Copy constructor.
    //! The new record owns its data since a record does not know whether
    //! or not it's OK to rely on the outside handle.
    vector_record(const vector_record& rec)
      : vector_numbering_ptr(rec.vector_numbering_ptr), vec_own(true),
        vec_ptr(new vec(*(rec.vec_ptr))) {
    }

    ~vector_record() {
      if (vec_own) {
        delete(vec_ptr);
      }
    }


    // Getters and helpers
    //==========================================================================

    //! Returns true iff this record has a value for this variable.
    bool has_variable(vector_variable* v) const;

    //! Returns the vector part of this record as an assignment.
    prl::vector_assignment vector_assignment() const;

    //! Converts this record to a vector assignment.
    operator prl::vector_assignment() const {
      return this->vector_assignment();
    }

    //! Returns the vector part of this record as an assignment,
    //! but only for the given variables X.
    //! @param X  All of these variables must be in this record.
    prl::vector_assignment assignment(const vector_domain& X) const;

    //! Returns the number of vector variables.
    size_t num_vector() const {
      return vector_numbering_ptr->size();
    }

    //! Returns list of vector variables in the natural order.
    //! NOTE: This is not stored and must be computed from vector_numbering!
    vector_var_vector vector_list() const;

    //! Returns the set of arguments.
    //! NOTE: This is not stored and must be computed from vector_numbering!
    vector_domain variables() const;

    //! Returns the vector component of this record as one continuous vector
    vec& vector() {
      return *vec_ptr;
    }

    //! Returns the vector component of this record as one continuous vector
    const vec& vector() const {
      return *vec_ptr;
    }

    //! Returns element i of the vector component of this record (when
    //!  represented as one continuous vector)
    //! Warning: The bounds are not checked!
    double& vector(size_t i) {
      return vec_ptr->operator[](i);
    }

    //! Returns element i of the vector component of this record (when
    //!  represented as one continuous vector)
    //! Warning: The bounds are not checked!
    double vector(size_t i) const {
      return vec_ptr->operator[](i);
    }

    //! Returns element j of the value of variable v in this record.
    //! Warning: The bounds are not checked!
    double& vector(vector_variable* v, size_t j) {
      size_t i(safe_get(*vector_numbering_ptr, v));
      return vec_ptr->operator[](i + j);
    }

    //! Returns the value of variable v in this record.
    //! Warning: The bounds are not checked!
    double vector(vector_variable* v, size_t j) const {
      size_t i(safe_get(*vector_numbering_ptr, v));
      return vec_ptr->operator[](i + j);
    }

    //! Sets the given vector vals to have the values of vars from this record.
    //! @param vals  Does not need to be pre-allocated.
    //! @param vars  All of these variables must have values in this record.
    void vector_values(vec& vals, const vector_var_vector& vars) const {
      size_t vars_size(vector_size(vars));
      if (vars_size != vals.size())
        vals.resize(vars_size);
      size_t i(0); // index into vals
      foreach(vector_variable* v, vars) {
        size_t j(safe_get(*vector_numbering_ptr, v));
        for (size_t k(0); k < v->size(); ++k) {
          vals[i] = vec_ptr->operator[](j + k);
          ++i;
        }
      }
    }

    //! Sets the given vector to the indices of the given variables' values
    //! in this record.
    void vector_indices(ivec& indices, const vector_var_vector& vars) const {
      size_t vars_size(vector_size(vars));
      if (indices.size() != vars_size)
        indices.resize(vars_size);
      size_t i(0); // index into indices
      foreach(vector_variable* v, vars) {
        size_t j(safe_get(*vector_numbering_ptr, v));
        for (size_t k(0); k < v->size(); ++k) {
          indices[i] = j + k;
          ++i;
        }
      }
    }

    //! Write the record to the given output stream.
    template <typename CharT, typename Traits>
    void write(std::basic_ostream<CharT, Traits>& out) const {
      out << this->vector_assignment();
    }

    /*
     * Set this record's view.
     * @param fin_vars  finite variables to be included in view
     * @param vec_vars  vector variables to be included in view
     * @param vec_components  vec_components[i] = components of i^th variable
     *                        in vec_vars to include in view; if none given,
     *                        then includes all
     */
    /*
    void set_view(var_vector fin_vars, var_vector vec_vars,
                  std::vector<std::vector<size_t> > vec_components
                  = std::vector<std::vector<size_t> >()) {
      fin_view.clear();
      foreach(variable_h v, fin_vars)
        fin_view.push_back(finite_numbering_ptr_->operator[v]);
      vec_view.clear();
      for (size_t j = 0; j < vec_vars.size(); ++j) {
        size_t start = vector_numbering_ptr_->operator[vec_vars[j]];
        if (vec_components.empty() || vec_components[j].empty())
          for (size_t k = 0; k < vec_vars[j].size(); ++k)
            vec_view.push_back(start + k);
        else
          foreach(size_t k, vec_components[j])
            vec_view.push_back(start + k);
      }
      is_view = true;
    }
    //! Resets the record's view to be complete.
    void reset_view() { is_view = false; }
    */

    // Mutating operations
    //==========================================================================

    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    vector_record& operator=(const vector_record& rec);

    //! Copies values from the given assignment (but only those for variables
    //! in this record.
    //! NOTE: The assignment must give values to all variables in this record.
    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    vector_record& operator=(const prl::vector_assignment& a);

    //! Clear the record.
    void clear();

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void
    reset(copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
          size_t vector_dim);

    //! Set vector data to be this value (stored in the record itself).
    void set_vector_val(const vec& val) {
      assert(vec_own);
      vec_ptr->operator=(val);
    }

    //! Set vector data to reference this value (stored outside of the record).
    void set_vector_ptr(vec* val) {
      assert(!vec_own);
      vec_ptr = val;
    }

  }; // class vector_record

  // Free functions
  //==========================================================================

  // @todo Fix this! (See symbolic_oracle.cpp)
  template <typename V, typename CharT, typename Traits>
  std::basic_ostream<CharT, Traits>&
  operator<<(std::basic_ostream<CharT, Traits>& out,
             const vector_record& r) {
    r.write(out);
    return out;
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_VECTOR_RECORD_HPP
