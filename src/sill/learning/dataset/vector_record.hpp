
#ifndef SILL_VECTOR_RECORD_HPP
#define SILL_VECTOR_RECORD_HPP

#include <map>

#include <sill/base/vector_assignment.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/copy_ptr.hpp>
#include <sill/learning/dataset/datasource_info_type.hpp>
#include <sill/math/linear_algebra_types.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  struct datasource_info_type;

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
   *
   * @todo Allow vector_records to be restricted to a set of variables.
   * @todo Change assignment(), finite_assignment(), etc. to allow efficient
   *       access for datasets whose native type is assignment.
   *
   * \ingroup learning_dataset
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class vector_record {

    // Public types and data members
    //==========================================================================
  public:

    //! Type of variable used.
    typedef vector_variable variable_type;

    //! Linear algebra specification
    typedef LA la_type;

    typedef typename la_type::vector_type vector_type;
    typedef typename la_type::value_type  value_type;

    //! Type of map value from a vector variable to the first index in a
    //! vector of values.
    typedef std::pair<vector_variable*, size_t> vector_var_index_pair;

    //! Map from vector variables to the first indices for those variables'
    //! values in a vector of values.
    copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr;

    //! True if record owns its data
    bool vec_own;

    //! Handle for vector variable values for record
    vector_type* vec_ptr;

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
        vec_own(true), vec_ptr(new vector_type()) {
    }

    //! Constructor for a vector_record which owns its data.
    //! @param vector_dim  Size of the vector data.
    vector_record
    (copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
     size_t vector_dim)
      : vector_numbering_ptr(vector_numbering_ptr), vec_own(true),
        vec_ptr(new vector_type(vector_dim)) {
    }

    //! Constructor for a vector_record which uses data from its creator
    vector_record
    (copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
     vector_type* vec_ptr)
      : vector_numbering_ptr(vector_numbering_ptr),
        vec_own(false), vec_ptr(vec_ptr) {
    }

    //! Constructor for a vector_record which owns its data.
    explicit vector_record(const vector_var_vector& vector_seq,
                           size_t vector_dim)
      : vector_numbering_ptr(new std::map<vector_variable*,size_t>()),
        vec_own(true), vec_ptr(new vector_type(vector_dim)) {
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
      vec_ptr = new vector_type(k);
    }

    //! Copy constructor.
    //! The new record owns its data since a record does not know whether
    //! or not it's OK to rely on the outside handle.
    vector_record(const vector_record& rec)
      : vector_numbering_ptr(rec.vector_numbering_ptr), vec_own(true),
        vec_ptr(new vector_type(*(rec.vec_ptr))) {
    }

    ~vector_record() {
      if (vec_own) {
        delete(vec_ptr);
      }
    }

    bool operator==(const vector_record& other) const {
      if (*vector_numbering_ptr == *(other.vector_numbering_ptr) &&
          vector() == other.vector())
        return true;
      else
        return false;
    }

    bool operator!=(const vector_record& other) const {
      return !operator==(other);
    }

    // Getters and helpers
    //==========================================================================

    //! Returns true iff this record has a value for this variable.
    bool has_variable(vector_variable* v) const {
      return (vector_numbering_ptr->count(v) != 0);
    }

    //! Returns the vector part of this record as an assignment.
    sill::vector_assignment vector_assignment() const {
      sill::vector_assignment a;
      foreach(const vector_var_index_pair& p, *vector_numbering_ptr) {
        vec v(p.first->size(), 0);
        for(size_t j = 0; j < p.first->size(); ++j)
          v[j] = vec_ptr->operator[](j+p.second);
        a[p.first] = v;
      }
      return a;
    }

    //! Converts this record to a vector assignment.
    operator sill::vector_assignment() const {
      return this->vector_assignment();
    }

    //! Returns the vector part of this record as an assignment,
    //! but only for the given variables X.
    //! @param X  All of these variables must be in this record.
    sill::vector_assignment assignment(const vector_domain& X) const {
      sill::vector_assignment a;
      foreach(vector_variable* v, X) {
        size_t v_index(safe_get(*vector_numbering_ptr, v));
        vector_type val(v->size());
        for(size_t j(0); j < v->size(); ++j)
          val[j] = vec_ptr->operator[](v_index + j);
        a[v] = val;
      }
      return a;
    }

    //! For the given variables X, add their values in this record to
    //! the given assignment.
    //! @param X  All of these variables must be in this record.
    void
    add_assignment(const vector_domain& X, sill::vector_assignment& a) const {
      foreach(vector_variable* v, X) {
        size_t v_index(safe_get(*vector_numbering_ptr, v));
        vector_type val(v->size());
        for(size_t j(0); j < v->size(); ++j)
          val[j] = vec_ptr->operator[](v_index + j);
        a[v] = val;
      }
    }

    //! Returns the number of vector variables.
    size_t num_vector() const {
      return vector_numbering_ptr->size();
    }

    //! Returns list of vector variables in the natural order.
    //! NOTE: This is not stored and must be computed from vector_numbering!
    vector_var_vector vector_list() const {
      vector_var_vector vlist(vector_numbering_ptr->size(), NULL);
      for (std::map<vector_variable*,size_t>::const_iterator
             it(vector_numbering_ptr->begin());
           it != vector_numbering_ptr->end(); ++it)
        vlist[it->second] = it->first;
      return vlist;
    }

    //! Returns the set of arguments.
    //! NOTE: This is not stored and must be computed from vector_numbering!
    vector_domain variables() const {
      return keys(*vector_numbering_ptr);
    }

    //! Returns the vector component of this record as one continuous vector
    vector_type& vector() {
      return *vec_ptr;
    }

    //! Returns the vector component of this record as one continuous vector
    const vector_type& vector() const {
      return *vec_ptr;
    }

    //! Returns element i of the vector component of this record (when
    //!  represented as one continuous vector)
    //! Warning: The bounds are not checked!
    value_type& vector(size_t i) {
      return vec_ptr->operator[](i);
    }

    //! Returns element i of the vector component of this record (when
    //!  represented as one continuous vector)
    //! Warning: The bounds are not checked!
    value_type vector(size_t i) const {
      return vec_ptr->operator[](i);
    }

    //! Returns element j of the value of variable v in this record.
    //! Warning: The bounds are not checked!
    value_type& vector(vector_variable* v, size_t j) {
      size_t i(safe_get(*vector_numbering_ptr, v));
      return vec_ptr->operator[](i + j);
    }

    //! Returns the value of variable v in this record.
    //! Warning: The bounds are not checked!
    value_type vector(vector_variable* v, size_t j) const {
      size_t i(safe_get(*vector_numbering_ptr, v));
      return vec_ptr->operator[](i + j);
    }

    //! Sets the given vector vals to have the values of vars from this record.
    //! @param vals  Does not need to be pre-allocated.
    //! @param vars  All of these variables must have values in this record.
    template <typename VecType>
    void vector_values(VecType& vals, const vector_var_vector& vars) const {
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
    template <typename IVecType>
    void
    vector_indices(IVecType& indices, const vector_var_vector& vars) const {
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
    void write(std::ostream& out) const {
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
    vector_record& operator=(const vector_record& rec) {
      vector_numbering_ptr = rec.vector_numbering_ptr;
      if (vec_own) {
        vec_ptr->resize(rec.vec_ptr->size());
        for (size_t j(0); j < rec.vec_ptr->size(); ++j)
          vec_ptr->operator[](j) = rec.vec_ptr->operator[](j);
      } else {
        vec_own = true;
        vec_ptr = new vector_type(*(rec.vec_ptr));
      }
      return *this;
    }

    //! Copies values from the given assignment (but only those for variables
    //! in this record.
    //! NOTE: The assignment must give values to all variables in this record.
    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    vector_record& operator=(const sill::vector_assignment& a) {
      size_t a_vars_size(0);
      foreach(const sill::vector_assignment::value_type& a_val, a) {
        a_vars_size += a_val.first->size();
      }
      if (!vec_own) {
        vec_ptr = new vector_type(a_vars_size, 0.);
        vec_own = true;
      } else {
        vec_ptr->resize(a_vars_size);
      }
      foreach(const vector_var_index_pair& p, *vector_numbering_ptr) {
        const vector_type& v = safe_get(a, p.first);
        for (size_t j = 0; j < p.first->size(); ++j)
          vec_ptr->operator[](p.second + j) = v[j];
      }
      return *this;
    }

    //! Clear the record.
    void clear() {
      vector_numbering_ptr->clear();
      if (vec_own) {
        vec_ptr->clear();
      } else {
        vec_own = true;
        vec_ptr = new vector_type();
      }
    }

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void
    reset(copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
          size_t vector_dim) {
      this->vector_numbering_ptr = vector_numbering_ptr;
      if (vec_own) {
        vec_ptr->resize(vector_dim);
      } else {
        vec_own = true;
        vec_ptr = new vector_type(vector_dim);
      }
    }

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void reset(const datasource_info_type& ds_info) {
      datasource_info_type::build_vector_numbering(ds_info.vector_seq,
                                                   *vector_numbering_ptr);
      size_t vdim = vector_size(ds_info.vector_seq);
      if (vec_own) {
        vec_ptr->resize(vdim);
      } else {
        vec_own = true;
        vec_ptr = new vector_type(vdim);
      }
    }

    //! Set vector data to be this value (stored in the record itself).
    void set_vector_val(const vector_type& val) {
      assert(vec_own);
      vec_ptr->operator=(val);
    }

    //! Set vector data to reference this value (stored outside of the record).
    void set_vector_ptr(vector_type* val) {
      assert(!vec_own);
      vec_ptr = val;
    }

    /**
     * For each variable in this record,
     * set the value according to the assignment,
     * with record variables mapped to assignment variables according to vmap.
     *
     * @param a     Assignment covering all variables in this record,
     *               modulo the variable mapping.
     * @param vmap  Variable mapping: Variables in record --> Variables in a.
     */
    void copy_assignment_mapped(const sill::vector_assignment& a,
                                const vector_var_map& vmap) {
      for (std::map<vector_variable*, size_t>::const_iterator it =
             vector_numbering_ptr->begin();
           it != vector_numbering_ptr->end();
           ++it) {
        vec_ptr->set_subvector
          (irange(it->second, it->second + it->first->size()),
           safe_get(a, safe_get(vmap, it->first)));
      }
    }

    /**
     * For each variable in this record,
     * set the value according to the given record,
     * with variables mapped according to vmap.
     *
     * @param r     Record covering all variables in this record,
     *               modulo the variable mapping.
     * @param vmap  Variable mapping: Vars in this record --> Vars in other.
     */
    void copy_record_mapped(const vector_record& r,
                            const vector_var_map& vmap) {
      for (std::map<vector_variable*, size_t>::const_iterator it =
             vector_numbering_ptr->begin();
           it != vector_numbering_ptr->end();
           ++it) {
        size_t i(safe_get(*(r.vector_numbering_ptr),
                          safe_get(vmap,it->first)));
        for (size_t j(it->second); j < it->second + it->first->size(); ++j) {
          this->vector(j) = r.vector(i);
          ++i;
        }
      }
    }

  }; // class vector_record

  // Free functions
  //==========================================================================

  // @todo Fix this! (See symbolic_oracle.cpp)
  template <typename LA>
  std::ostream& operator<<(std::ostream& out, const vector_record<LA>& r) {
    r.write(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_VECTOR_RECORD_HPP
