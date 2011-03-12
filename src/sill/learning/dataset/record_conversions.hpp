
#ifndef SILL_RECORD_CONVERSIONS_HPP
#define SILL_RECORD_CONVERSIONS_HPP

#include <sill/base/assignment.hpp>
#include <sill/math/linear_algebra_types.hpp>
//#include <sill/learning/dataset/record.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Sets the vals vector to be the values for the variables vars
   * in the given record.
   */
  /*
  template <linear_algebra_enum LA>
  void
  vector_record2vector
  (const vector_record<LA>& r,
   const vector_var_vector& vars,
   typename linear_algebra_types<double,size_t,LA>::vector_type& vals);
  */

  /**
   * Adds the given vector x matching vector variables X to the assignment.
   */
  template <typename VecType>
  void add_vector2vector_assignment(const vector_var_vector& X,
                                    const VecType& x,
                                    vector_assignment& va);

  /**
   * Converts the given finite assignment into finite record data.
   * @param fa          Source data.
   * @param finite_seq  Finite variable ordering used for findata.
   * @param findata     Target data.
   */
  void
  finite_assignment2vector(const finite_assignment& fa,
                           const finite_var_vector& finite_seq,
                           std::vector<size_t>& findata);

  /**
   * Converts the given vector assignment into vector record data.
   * @param fa          Source data.
   * @param vector_seq  Vector variable ordering used for vecdata.
   * @param vecdata     Target data.
   */
  template <typename VecType>
  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           VecType& vecdata);

  /**
   * Fills the data in a record with data from the given assignment,
   * using the given variable mapping.
   * @param r  (Return value) Record.
   * @param a  Assignment.
   * @param vmap  Variable mapping: Variables in r --> Variables in a.
   */
  /*
  template <linear_algebra_enum LA>
  void fill_record_with_assignment(record<LA>& r, const assignment& a,
                                   const var_map& vmap);
  */

  /**
   * Fills the data in a record with data from the given assignment,
   * using the given variable mapping.
   * @param r  (Return value) Record.
   * @param a  Assignment.
   * @param vmap  Variable mapping: Variables in r --> Variables in a.
   */
  /*
  void fill_record_with_assignment(finite_record& r, const finite_assignment& a,
                                   const finite_var_map& vmap);
  */

  /**
   * Fills the data in a record with data from the given assignment,
   * using the given variable mapping.
   * @param r  (Return value) Record.
   * @param a  Assignment.
   * @param vmap  Variable mapping: Variables in r --> Variables in a.
   */
  /*
  template <linear_algebra_enum LA>
  void fill_record_with_assignment(vector_record<LA>& r,
                                   const vector_assignment& a,
                                   const vector_var_map& vmap);
  */

  /**
   * Fills the data in a record with data from another record,
   * using the given variable mapping.
   * @param to    (Return value) Record.
   * @param from  Record.
   * @param vmap  Variable mapping: Variables in 'to' --> Variables in 'from'.
   */
  /*
  template <linear_algebra_enum LA>
  void fill_record_with_record(record<LA>& to, const record<LA>& from,
                               const var_map& vmap);
  */

  /**
   * Fills the data in a record with data from another record,
   * using the given variable mapping.
   * @param to    (Return value) Record.
   * @param from  Record.
   * @param vmap  Variable mapping: Variables in 'to' --> Variables in 'from'.
   */
  /*
  void fill_record_with_record(finite_record& to, const finite_record& from,
                               const finite_var_map& vmap);
  */

  /**
   * Fills the data in a record with data from another record,
   * using the given variable mapping.
   * @param to    (Return value) Record.
   * @param from  Record.
   * @param vmap  Variable mapping: Variables in 'to' --> Variables in 'from'.
   */
  /*
  template <linear_algebra_enum LA>
  void fill_record_with_record(vector_record<LA>& to,
                               const vector_record<LA>& from,
                               const vector_var_map& vmap);
  */

  //============================================================================
  // Implementations of above functions
  //============================================================================

  /*
  template <linear_algebra_enum LA>
  void
  vector_record2vector
  (const vector_record<LA>& r,
   const vector_var_vector& vars,
   typename linear_algebra_types<double,size_t,LA>::vector_type& vals) {
    vals.resize(vector_size(vars));
    size_t i(0); // index into vals
    foreach(vector_variable* v, vars) {
      for (size_t j(0); j < v->size(); ++j) {
        vals[i] = r.vector(v, j);
        ++i;
      }
    }
  }
  */

  template <typename VecType>
  void add_vector2vector_assignment(const vector_var_vector& X,
                                    const VecType& x,
                                    vector_assignment& va) {
    size_t i = 0; // index into x
    vec val;
    foreach(vector_variable* v, X) {
      val.resize(v->size());
      if (i + v->size() > x.size()) {
        throw std::invalid_argument
          (std::string("add_vector2vector_assignment(X,x,va)") +
           " given X,x of non-matching dimensionalities (|X| > |x|).");
      }
      for (size_t j = 0; j < v->size(); ++j)
        val[j] = x[i++];
      va[v] = val;
    }
    if (i != x.size()) {
        throw std::invalid_argument
          (std::string("add_vector2vector_assignment(X,x,va)") +
           " given X,x of non-matching dimensionalities (|X| < |x|).");
    }
  }

  template <typename VecType>
  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           VecType& vecdata) {
    vecdata.resize(vector_size(vector_seq));
    size_t k(0); // index into vecdata
    vector_assignment::const_iterator va_end = va.end();
    foreach(vector_variable* v, vector_seq) {
      vector_assignment::const_iterator it(va.find(v));
      if (it == va_end) {
        throw std::runtime_error("vector_assignment2vector given vector_seq with variables not appearing in given assignment.");
      }
      const vec& tmpvec = it->second;
      for (size_t j(0); j < v->size(); j++)
        vecdata[k + j] = tmpvec[j];
      k += v->size();
    }
  }

  /*
  template <linear_algebra_enum LA>
  void fill_record_with_assignment(record<LA>& r, const assignment& a,
                                   const var_map& vmap) {
    for (std::map<finite_variable*, size_t>::const_iterator it =
           r.finite_numbering_ptr->begin();
         it != r.finite_numbering_ptr->end();
         ++it) {
      r.finite(it->second) =
        safe_get(a.finite(),
                 (finite_variable*)(safe_get(vmap,(variable*)(it->first))));
    }
    for (std::map<vector_variable*, size_t>::const_iterator it =
           r.vector_numbering_ptr->begin();
         it != r.vector_numbering_ptr->end();
         ++it) {
      r.vector().set_subvector
        (irange(it->second, it->second + it->first->size()),
         safe_get(a.vector(),
                  (vector_variable*)(safe_get(vmap, (variable*)(it->first)))));
    }
  }
  */

  /*
  template <linear_algebra_enum LA>
  void fill_record_with_assignment(vector_record<LA>& r,
                                   const vector_assignment& a,
                                   const vector_var_map& vmap) {
    for (std::map<vector_variable*, size_t>::const_iterator it =
           r.vector_numbering_ptr->begin();
         it != r.vector_numbering_ptr->end();
         ++it) {
      r.vector().set_subvector
        (irange(it->second, it->second + it->first->size()),
         safe_get(a, safe_get(vmap, it->first)));
    }
  }
  */

  /*
  template <linear_algebra_enum LA>
  void fill_record_with_record(record<LA>& to, const record<LA>& from,
                               const var_map& vmap) {
    for (std::map<finite_variable*, size_t>::const_iterator it =
           to.finite_numbering_ptr->begin();
         it != to.finite_numbering_ptr->end();
         ++it) {
      to.finite(it->second) =
        from.finite((finite_variable*)(safe_get(vmap,(variable*)(it->first))));
    }
    for (std::map<vector_variable*, size_t>::const_iterator it =
           to.vector_numbering_ptr->begin();
         it != to.vector_numbering_ptr->end();
         ++it) {
      size_t i =
        safe_get(*(from.vector_numbering_ptr),
                 (vector_variable*)(safe_get(vmap,(variable*)(it->first))));
      for (size_t j(it->second); j < it->second + it->first->size(); ++j) {
        to.vector(j) = from.vector(i);
        ++i;
      }
    }
  }
  */

  /*
  template <linear_algebra_enum LA>
  void fill_record_with_record(vector_record<LA>& to,
                               const vector_record<LA>& from,
                               const vector_var_map& vmap) {
    for (std::map<vector_variable*, size_t>::const_iterator it =
           to.vector_numbering_ptr->begin();
         it != to.vector_numbering_ptr->end();
         ++it) {
      size_t i(safe_get(*(from.vector_numbering_ptr),
                        safe_get(vmap,it->first)));
      for (size_t j(it->second); j < it->second + it->first->size(); ++j) {
        to.vector(j) = from.vector(i);
        ++i;
      }
    }
  }
  */

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RECORD_CONVERSIONS_HPP

