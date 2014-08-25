#ifndef SILL_RECORD_HPP
#define SILL_RECORD_HPP

#include <sill/base/assignment.hpp>
#include <sill/base/variables.hpp>
#include <sill/learning/dataset_old/finite_record.hpp>
#include <sill/learning/dataset_old/vector_record.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  finite_var_vector extract_finite_var_vector(const var_vector& vars);
  vector_var_vector extract_vector_var_vector(const var_vector& vars);

  /**
   * A type that provides a mutable view of a single datapoint.
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
   * \todo Allow records to be restricted to a set of variables.
   * @todo Change assignment(), finite_assignment(), etc. to allow efficient
   *       access for datasets whose native type is assignment.
   * \ingroup learning_dataset
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<double,size_t>)
   */
  template <typename LA = dense_linear_algebra<> >
  class record
    : public finite_record_old, public vector_record_old<LA> {

    // Public types and data members
    //==========================================================================
  public:

    //! Type of variable used.
    typedef variable variable_type;

    typedef typename vector_record_old<LA>::la_type la_type;
    typedef typename la_type::vector_type vector_type;
    typedef typename la_type::value_type  value_type;

    using vector_record_old<la_type>::vector_numbering_ptr;
    using vector_record_old<la_type>::vec_own;
    using vector_record_old<la_type>::vec_ptr;

    // Constructors
    //==========================================================================

    //! Constructs an empty record which owns its data.
    record() : finite_record_old(), vector_record_old<la_type>() {
    }

    //! Constructor for a record which owns its data.
    record(copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr,
           copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
           size_t vector_dim)
      : finite_record_old(finite_numbering_ptr),
        vector_record_old<la_type>(vector_numbering_ptr, vector_dim) {
    }

    //! Constructor for a record which uses data from its creator
    record(copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr,
           copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
           std::vector<size_t>* fin_ptr, vector_type* vec_ptr)
      : finite_record_old(finite_numbering_ptr, fin_ptr),
        vector_record_old<la_type>(vector_numbering_ptr, vec_ptr) {
    }

    //! Constructor for a record which owns its data.
    record(const finite_var_vector& finite_seq,
           const vector_var_vector& vector_seq,
           size_t vector_dim)
      : finite_record_old(finite_seq), vector_record_old<la_type>(vector_seq, vector_dim) {
    }

    //! Constructor for a record with only finite data which owns its data.
    explicit record(const finite_record_old& r)
      : finite_record_old(r), vector_record_old<la_type>() {
    }

    //! Constructor for a record with only finite data which owns its data.
    record(const finite_var_vector& finite_seq)
      : finite_record_old(finite_seq), vector_record_old<la_type>() {
    }

    //! Constructor for a record with only vector data which owns its data.
    explicit record(const vector_record_old<la_type>& r)
      : finite_record_old(), vector_record_old<la_type>(r) {
    }

    //! Constructor for a record with only vector data which owns its data.
    record(const vector_var_vector& vector_seq)
      : finite_record_old(), vector_record_old<la_type>(vector_seq) {
    }

    //! Constructor for a record which owns its data.
    record(const var_vector& var_seq)
      : finite_record_old(extract_finite_var_vector(var_seq)),
        vector_record_old<la_type>(extract_vector_var_vector(var_seq)) {
    }

    //! Constructor for a record which owns its data.
    //! @param ds_info  Gives finite and vector sequences.
    record(const datasource_info_type& ds_info)
      : finite_record_old(ds_info.finite_seq),
        vector_record_old<la_type>(ds_info.vector_seq, vector_size(ds_info.vector_seq)) {
    }

    //! Copy constructor.
    //! The new record owns its data since a record does not know whether
    //! or not it's OK to rely on the outside handle.
    record(const record& rec) : finite_record_old(rec), vector_record_old<la_type>(rec) {
    }

    bool operator==(const record& other) const {
      if (finite_record_old::operator==(other) &&
          vector_record_old<la_type>::operator==(other))
        return true;
      else
        return false;
    }

    bool operator!=(const record& other) const {
      return !operator==(other);
    }

    // Getters and helpers
    //==========================================================================

    //! Returns this record as an assignment.
    sill::assignment assignment() const {
      return sill::assignment(this->finite_assignment(),
                             this->vector_assignment());
    }

    //! Converts this record to an assignment.
    operator sill::assignment() const {
      return this->assignment();
    }

    //! Returns this record as an assignment,
    //! but only for the given variables X.
    //! @param X  All of these variables must be in this record.
    sill::assignment assignment(const domain& X) const;

    //! Returns the finite part of this record as an assignment,
    //! but only for the given variables X.
    //! @param X  All of these variables must be in this record.
    sill::finite_assignment assignment(const finite_domain& X) const {
      return finite_record_old::assignment(X);
    }

    //! Returns the vector part of this record as an assignment,
    //! but only for the given variables X.
    //! @param X  All of these variables must be in this record.
    sill::vector_assignment assignment(const vector_domain& X) const {
      return vector_record_old<la_type>::assignment(X);
    }

    //! For the given variables X, add their values in this record to
    //! the given assignment.
    //! @param X  All of these variables must be in this record.
    void
    add_to_assignment(const domain& X, sill::assignment& a) const;

    //! For the given variables X, add their values in this record to
    //! the given assignment.
    //! @param X  All of these variables must be in this record.
    void
    add_to_assignment(const finite_domain& X,
                      sill::finite_assignment& a) const {
      finite_record_old::add_to_assignment(X, a);
    }

    //! For the given variables X, add their values in this record to
    //! the given assignment.
    //! @param X  All of these variables must be in this record.
    void
    add_to_assignment(const vector_domain& X,
                      sill::vector_assignment& a) const {
      vector_record_old<la_type>::add_to_assignment(X, a);
    }

    //! Returns the set of arguments.
    //! NOTE: This is not stored and must be computed!
    domain variables() const {
      finite_domain fvars(this->finite_record_old::variables());
      vector_domain vvars(this->vector_record_old<la_type>::variables());
      domain vars;
      vars.insert(fvars.begin(), fvars.end());
      vars.insert(vvars.begin(), vvars.end());
      return vars;
    }

    //! Returns the number of arguments.
    size_t num_variables() const {
      return finite_record_old::num_variables()
        + vector_record_old<la_type>::num_variables();
    }

    //! Write the record to the given output stream.
    void write(std::ostream& out) const {
      out << this->assignment();
    }

    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-finite component.
    record& operator=(const finite_record_old& rec) {
      vector_record_old<la_type>::clear();
      finite_record_old::operator=(rec);
      return *this;
    }

    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-vector component.
    record& operator=(const vector_record_old<la_type>& rec) {
      finite_record_old::clear();
      vector_record_old<la_type>::operator=(rec);
      return *this;
    }

    //! Copies values from the given assignment (but only those for variables
    //! in this record.
    //! NOTE: The assignment must give values to all variables in this record.
    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-finite component.
    record& operator=(const sill::finite_assignment& a) {
      vector_record_old<la_type>::clear();
      finite_record_old::operator=(a);
      return *this;
    }

    //! Copies values from the given assignment (but only those for variables
    //! in this record.
    //! NOTE: The assignment must give values to all variables in this record.
    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-vector component.
    record& operator=(const sill::vector_assignment& a) {
      finite_record_old::clear();
      vector_record_old<la_type>::operator=(a);
      return *this;
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

    //! Clear the record.
    void clear() {
      finite_record_old::clear();
      vector_record_old<la_type>::clear();
    }

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void
    reset(copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr,
          copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
          size_t vector_dim) {
      finite_record_old::reset(finite_numbering_ptr);
      vector_record_old<la_type>::reset(vector_numbering_ptr, vector_dim);
    }

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void reset(const datasource_info_type& ds_info) {
      finite_record_old::reset(ds_info);
      vector_record_old<la_type>::reset(ds_info);
    }

    /**
     * For each variable appearing in BOTH the given assignment and this record,
     * set this record's value to match the assignment's value.
     * This does not change this record's domain.
     */
    void copy_from_assignment(const sill::assignment& a) {
      finite_record_old::copy_from_assignment(a.finite());
      vector_record_old<la_type>::copy_from_assignment(a.vector());
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
    void copy_from_assignment_mapped(const sill::assignment& a,
                                     const var_map& vmap) {
      for (std::map<finite_variable*, size_t>::const_iterator it =
             finite_numbering_ptr->begin();
           it != finite_numbering_ptr->end();
           ++it) {
        this->finite(it->second) =
          safe_get(a.finite(),
                   (finite_variable*)(safe_get(vmap,(variable*)(it->first))));
      }
      for (std::map<vector_variable*, size_t>::const_iterator it =
             vector_numbering_ptr->begin();
           it != vector_numbering_ptr->end();
           ++it) {
        this->vector().set_subvector
          (span(it->second, it->second + it->first->size() - 1),
           safe_get(a.vector(),
                    (vector_variable*)(safe_get(vmap,(variable*)(it->first)))));
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
    void copy_from_record_mapped(const record& r,
                                 const var_map& vmap) {
      for (std::map<finite_variable*, size_t>::const_iterator it =
             finite_numbering_ptr->begin();
           it != finite_numbering_ptr->end();
           ++it) {
        this->finite(it->second) =
          r.finite((finite_variable*)(safe_get(vmap,(variable*)(it->first))));
      }
      for (std::map<vector_variable*, size_t>::const_iterator it =
             vector_numbering_ptr->begin();
           it != vector_numbering_ptr->end();
           ++it) {
        size_t i =
          safe_get(*(r.vector_numbering_ptr),
                   (vector_variable*)(safe_get(vmap,(variable*)(it->first))));
        for (size_t j(it->second); j < it->second + it->first->size(); ++j) {
          this->vector(j) = r.vector(i);
          ++i;
        }
      }
    }

  }; // class record

  // Free functions
  //==========================================================================

  // @todo Fix this! (See symbolic_oracle.cpp)
  template <typename LA>
  std::ostream& operator<<(std::ostream& out, const record<LA>& r) {
    r.write(out);
    return out;
  }

  //============================================================================
  // Implementations of methods in record
  //============================================================================

  // Getters and helpers
  //==========================================================================

  template <typename LA>
  sill::assignment record<LA>::assignment(const domain& X) const {
    sill::assignment a;
    foreach(variable* v, X) {
      switch(v->type()) {
      case variable::FINITE_VARIABLE:
        {
          finite_variable* vf = dynamic_cast<finite_variable*>(v);
          a.finite()[vf] = this->finite(vf);
        }
        break;
      case variable::VECTOR_VARIABLE:
        {
          vector_variable* vv = dynamic_cast<vector_variable*>(v);
          size_t v_index(safe_get(*vector_numbering_ptr, vv));
          vec val(vv->size());
          for(size_t j(0); j < vv->size(); ++j)
            val[j] = vec_ptr->operator[](v_index + j);
          a.vector()[vv] = val;
        }
        break;
      default:
        assert(false);
      }
    }
    return a;
  }

  template <typename LA>
  void
  record<LA>::add_to_assignment(const domain& X, sill::assignment& a) const {
    foreach(variable* v, X) {
      switch(v->type()) {
      case variable::FINITE_VARIABLE:
        {
          finite_variable* vf = dynamic_cast<finite_variable*>(v);
          a.finite()[vf] = this->finite(vf);
        }
        break;
      case variable::VECTOR_VARIABLE:
        {
          vector_variable* vv = dynamic_cast<vector_variable*>(v);
          size_t v_index(safe_get(*vector_numbering_ptr, vv));
          vec val(vv->size());
          for(size_t j(0); j < vv->size(); ++j)
            val[j] = vec_ptr->operator[](v_index + j);
          a.vector()[vv] = val;
        }
        break;
      default:
        assert(false);
      }
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_RECORD_HPP
