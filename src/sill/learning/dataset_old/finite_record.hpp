#ifndef SILL_FINITE_RECORD_OLD_HPP
#define SILL_FINITE_RECORD_OLD_HPP

#include <map>

#include <sill/base/finite_assignment.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/copy_ptr.hpp>
#include <sill/learning/dataset_old/datasource_info_type.hpp>
#include <sill/learning/dataset_old/finite_record_iterator.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  struct datasource_info_type;
  class finite_record_old_iterator;

  /**
   * A type that provides a mutable view of a single datapoint's finite data.
   * This provides flexible access to data in 2 forms:
   *  - vector form: a vector of values (for algorithms which know the
   *                 ordering of variables)
   *  - assignment form: a mapping from variables to values
   *
   * Records can store their own data but can also hold pointers into
   * data stored in other structures.
   * When a record is copied, the copy will store its own data.
   *
   * \author Joseph Bradley, Stanislav Funiak
   * \todo Allow finite_record_olds to be restricted to a set of variables.
   * @todo Change assignment(), finite_assignment(), etc. to allow efficient
   *       access for datasets whose native type is assignment.
   * \ingroup learning_dataset
   */
  class finite_record_old {

    // Public types and data members
    //==========================================================================
  public:

    //! Type of variable used.
    typedef finite_variable variable_type;

    //! Type of map value from a finite variable to its index
    typedef std::pair<finite_variable*, size_t> finite_var_index_pair;

    //! Map from finite variables to their indices.
    copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr;

    //! True if record owns its data
    bool fin_own;

    //! Handle for finite variable values for record
    std::vector<size_t>* fin_ptr;

    /*
// TODO: FINISH RECORD VIEWS
    //! Indicates if this is a view.
    bool is_fin_view;
    //! Indices of finite variables (elements in fin_ptr) in view.
    std::vector<size_t> fin_view;
    */

    // Constructors
    //==========================================================================

    //! Constructs an empty finite_record_old which owns its data.
    finite_record_old()
      : finite_numbering_ptr(new std::map<finite_variable*, size_t>()),
        fin_own(true), fin_ptr(new std::vector<size_t>()) {
    }

    //! Constructor for a finite_record_old which owns its data.
    explicit finite_record_old
    (copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr)
      : finite_numbering_ptr(finite_numbering_ptr), fin_own(true),
        fin_ptr(new std::vector<size_t>(finite_numbering_ptr->size())) {
    }

    //! Constructor for a finite_record_old which uses data from its creator
    finite_record_old
    (copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr,
     std::vector<size_t>* fin_ptr)
      : finite_numbering_ptr(finite_numbering_ptr),
        fin_own(false), fin_ptr(fin_ptr) {
    }

    //! Constructor for a finite_record_old which owns its data.
    explicit finite_record_old(const finite_var_vector& finite_seq)
      : finite_numbering_ptr(new std::map<finite_variable*,size_t>()),
        fin_own(true), fin_ptr(new std::vector<size_t>(finite_seq.size())) {
      for (size_t j(0); j < finite_seq.size(); ++j)
        finite_numbering_ptr->operator[](finite_seq[j]) = j;
    }

    //! Copy constructor.
    //! The new record owns its data since a record does not know whether
    //! or not it's OK to rely on the outside handle.
    finite_record_old(const finite_record_old& rec)
      : finite_numbering_ptr(rec.finite_numbering_ptr),
        fin_own(true), fin_ptr(new std::vector<size_t>(*(rec.fin_ptr))) {
    }

    ~finite_record_old() {
      if (fin_own) {
        delete(fin_ptr);
      }
    }

    bool operator==(const finite_record_old& other) const;

    bool operator!=(const finite_record_old& other) const;

    // Getters and helpers
    //==========================================================================

    //! Returns true iff this record has a value for this variable.
    bool has_variable(finite_variable* v) const;

    //! Returns the finite part of this record as an assignment.
    sill::finite_assignment finite_assignment() const;

    //! Converts this record to a finite assignment.
    operator sill::finite_assignment() const {
      return this->finite_assignment();
    }

    //! Returns the finite part of this record as an assignment,
    //! but only for the given variables X.
    //! @param X  All of these variables must be in this record.
    sill::finite_assignment assignment(const finite_domain& X) const;

    //! For the given variables X, add their values in this record to
    //! the given assignment.
    //! @param X  All of these variables must be in this record.
    void
    add_to_assignment(const finite_domain& X, sill::finite_assignment& a) const;

    //! Returns the number of finite variables.
    size_t num_finite() const {
      return finite_numbering_ptr->size();
    }

    //! Returns list of finite variables in the natural order.
    //! NOTE: This is not stored and must be computed from finite_numbering!
    finite_var_vector finite_list() const;

    //! Returns the set of arguments.
    //! NOTE: This is not stored and must be computed from vector_numbering!
    finite_domain variables() const;

    //! Returns the number of arguments.
    size_t num_variables() const;

    //! Returns the finite component of this record as a vector
    std::vector<size_t>& finite() {
      return *fin_ptr;
    }

    //! Returns the finite component of this record as a vector
    const std::vector<size_t>& finite() const {
      return *fin_ptr;
    }

    //! Returns the index in this record for variable v.
    size_t index(finite_variable* v) const {
      return safe_get(*finite_numbering_ptr, v);
    }

    //! Returns element i of the finite component of this record
    //! Warning: The bounds are not checked!
    size_t& finite(size_t i) {
      return fin_ptr->operator[](i);
    }

    //! Returns element i of the finite component of this record
    //! Warning: The bounds are not checked!
    size_t finite(size_t i) const {
      return fin_ptr->operator[](i);
    }

    //! Returns the value of variable v in this record.
    size_t& finite(finite_variable* v) {
      return fin_ptr->operator[](index(v));
    }

    //! Returns the value of variable v in this record.
    size_t finite(finite_variable* v) const {
      return fin_ptr->operator[](index(v));
    }

    //! (Similar to finite_assignment::find)
    //! Returns an iterator to the <variable, value> pair,
    //! or an end iterator if the variable is not found in this record.
    finite_record_old_iterator find(finite_variable* v) const;

    //! (Similar to finite_assignment::end)
    //! Returns an end iterator, to complement the find() method.
    finite_record_old_iterator end() const;

    //! Write the record to the given output stream.
    template <typename CharT, typename Traits>
    void write(std::basic_ostream<CharT, Traits>& out) const {
      out << this->finite_assignment();
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
    finite_record_old& operator=(const finite_record_old& rec);

    //! Copies values from the given assignment (but only those for variables
    //! in this record.
    //! NOTE: The assignment must give values to all variables in this record.
    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    finite_record_old& operator=(const sill::finite_assignment& a);

    //! Clear the record.
    void clear();

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void
    reset(copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr);

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void reset(const datasource_info_type& ds_info);

    //! Set finite data to be this value (stored in the record itself).
    void set_finite_val(const std::vector<size_t>& val);

    //! Set finite data to reference this value (stored outside of the record).
    void set_finite_ptr(std::vector<size_t>* val);

    /**
     * For each variable appearing in BOTH the given assignment and this record,
     * set this record's value to match the assignment's value.
     * This does not change this record's domain.
     */
    void copy_from_assignment(const sill::finite_assignment& a);

    /**
     * For each variable in this record,
     * set the value according to the assignment,
     * with record variables mapped to assignment variables according to vmap.
     *
     * @param a     Assignment covering all variables in this record,
     *               modulo the variable mapping.
     * @param vmap  Variable mapping: Variables in record --> Variables in a.
     */
    void copy_from_assignment_mapped(const sill::finite_assignment& a,
                                     const finite_var_map& vmap);

    /**
     * For each variable in this record,
     * set the value according to the given record,
     * with variables mapped according to vmap.
     *
     * @param r     Record covering all variables in this record,
     *               modulo the variable mapping.
     * @param vmap  Variable mapping: Vars in this record --> Vars in other.
     */
    void copy_from_record_mapped(const finite_record_old& r,
                                 const finite_var_map& vmap);

  }; // class finite_record_old

  // Free functions
  //==========================================================================

  // @todo Fix this! (See symbolic_oracle.cpp)
  std::ostream&
  operator<<(std::ostream& out, const finite_record_old& r);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_FINITE_RECORD_HPP
