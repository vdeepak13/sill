
#ifndef SILL_RECORD_HPP
#define SILL_RECORD_HPP

#include <map>

#include <sill/base/assignment.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/base/variables.hpp>
#include <sill/copy_ptr.hpp>
#include <sill/learning/dataset/datasource.hpp>
#include <sill/learning/dataset/finite_record.hpp>
#include <sill/learning/dataset/vector_record.hpp>
#include <sill/math/vector.hpp>

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
   */
  class record
    : public finite_record, public vector_record {

    // Public types and data members
    //==========================================================================
  public:

    //! Type of variable used.
    typedef variable variable_type;

    // Constructors
    //==========================================================================

    //! Constructs an empty record which owns its data.
    record() : finite_record(), vector_record() {
    }

    //! Constructor for a record which owns its data.
    record(copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr,
           copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
           size_t vector_dim)
      : finite_record(finite_numbering_ptr),
        vector_record(vector_numbering_ptr, vector_dim) {
    }

    //! Constructor for a record which uses data from its creator
    record(copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr,
           copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
           std::vector<size_t>* fin_ptr, vec* vec_ptr)
      : finite_record(finite_numbering_ptr, fin_ptr),
        vector_record(vector_numbering_ptr, vec_ptr) {
    }

    //! Constructor for a record which owns its data.
    record(const finite_var_vector& finite_seq,
           const vector_var_vector& vector_seq,
           size_t vector_dim)
      : finite_record(finite_seq), vector_record(vector_seq, vector_dim) {
    }

    //! Constructor for a record with only finite data which owns its data.
    explicit record(const finite_record& r)
      : finite_record(r), vector_record() {
    }

    //! Constructor for a record with only finite data which owns its data.
    record(const finite_var_vector& finite_seq)
      : finite_record(finite_seq), vector_record() {
    }

    //! Constructor for a record with only vector data which owns its data.
    explicit record(const vector_record& r)
      : finite_record(), vector_record(r) {
    }

    //! Constructor for a record with only vector data which owns its data.
    record(const vector_var_vector& vector_seq)
      : finite_record(), vector_record(vector_seq) {
    }

    //! Constructor for a record which owns its data.
    record(const var_vector& var_seq)
      : finite_record(extract_finite_var_vector(var_seq)),
        vector_record(extract_vector_var_vector(var_seq)) {
    }

    //! Constructor for a record which owns its data.
    //! @param ds_info  Gives finite and vector sequences.
    record(const datasource_info_type& ds_info)
      : finite_record(ds_info.finite_seq),
        vector_record(ds_info.vector_seq, vector_size(ds_info.vector_seq)) {
    }

    //! Copy constructor.
    //! The new record owns its data since a record does not know whether
    //! or not it's OK to rely on the outside handle.
    record(const record& rec) : finite_record(rec), vector_record(rec) {
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
    sill::finite_assignment assignment(const finite_domain& X) const;

    //! Returns the vector part of this record as an assignment,
    //! but only for the given variables X.
    //! @param X  All of these variables must be in this record.
    sill::vector_assignment assignment(const vector_domain& X) const;

    //! Write the record to the given output stream.
    template <typename CharT, typename Traits>
    void write(std::basic_ostream<CharT, Traits>& out) const {
      out << this->assignment();
    }

    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-finite component.
    record& operator=(const finite_record& rec) {
      vector_record::clear();
      finite_record::operator=(rec);
      return *this;
    }

    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-vector component.
    record& operator=(const vector_record& rec) {
      finite_record::clear();
      vector_record::operator=(rec);
      return *this;
    }

    //! Copies values from the given assignment (but only those for variables
    //! in this record.
    //! NOTE: The assignment must give values to all variables in this record.
    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-finite component.
    record& operator=(const sill::finite_assignment& a) {
      vector_record::clear();
      finite_record::operator=(a);
      return *this;
    }

    //! Copies values from the given assignment (but only those for variables
    //! in this record.
    //! NOTE: The assignment must give values to all variables in this record.
    //! The new copy owns its data since a record does not know whether
    //! or not it's OK to rely on the outside reference.
    //! This clears the non-vector component.
    record& operator=(const sill::vector_assignment& a) {
      finite_record::clear();
      vector_record::operator=(a);
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
      finite_record::clear();
      vector_record::clear();
    }

    //! Clears the stored data (if it is owned by the record)
    //! and resets the record (like a constructor).
    void
    reset(copy_ptr<std::map<finite_variable*, size_t> > finite_numbering_ptr,
          copy_ptr<std::map<vector_variable*, size_t> > vector_numbering_ptr,
          size_t vector_dim) {
      finite_record::reset(finite_numbering_ptr);
      vector_record::reset(vector_numbering_ptr, vector_dim);
    }

  }; // class record

  // Free functions
  //==========================================================================

  // @todo Fix this! (See symbolic_oracle.cpp)
  template <typename V, typename CharT, typename Traits>
  std::basic_ostream<CharT, Traits>&
  operator<<(std::basic_ostream<CharT, Traits>& out,
             const record& r) {
    r.write(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_RECORD_HPP
