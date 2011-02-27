
#ifndef SILL_RECORD_CONVERSIONS_HPP
#define SILL_RECORD_CONVERSIONS_HPP

#include <sill/base/assignment.hpp>
#include <sill/learning/dataset/record.hpp>

namespace sill {

  /**
   * Sets the vals vector to be the values for the variables vars
   * in the given record.
   */
  void
  vector_record2vector(const vector_record& r, const vector_var_vector& vars,
                       vec& vals);

  /**
   * Adds the given vector x matching vector variables X to the assignment.
   */
  void add_vector2vector_assignment(const vector_var_vector& X, const vec& x,
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
  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           vec& vecdata);

  /**
   * Fills the data in a record with data from the given assignment,
   * using the given variable mapping.
   * @param r  (Return value) Record.
   * @param a  Assignment.
   * @param vmap  Variable mapping: Variables in r --> Variables in a.
   */
  void fill_record_with_assignment(record& r, const assignment& a,
                                   const var_map& vmap);

  /**
   * Fills the data in a record with data from the given assignment,
   * using the given variable mapping.
   * @param r  (Return value) Record.
   * @param a  Assignment.
   * @param vmap  Variable mapping: Variables in r --> Variables in a.
   */
  void fill_record_with_assignment(finite_record& r, const finite_assignment& a,
                                   const finite_var_map& vmap);

  /**
   * Fills the data in a record with data from the given assignment,
   * using the given variable mapping.
   * @param r  (Return value) Record.
   * @param a  Assignment.
   * @param vmap  Variable mapping: Variables in r --> Variables in a.
   */
  void fill_record_with_assignment(vector_record& r, const vector_assignment& a,
                                   const vector_var_map& vmap);

  /**
   * Fills the data in a record with data from another record,
   * using the given variable mapping.
   * @param to    (Return value) Record.
   * @param from  Record.
   * @param vmap  Variable mapping: Variables in 'to' --> Variables in 'from'.
   */
  void fill_record_with_record(record& to, const record& from,
                               const var_map& vmap);

  /**
   * Fills the data in a record with data from another record,
   * using the given variable mapping.
   * @param to    (Return value) Record.
   * @param from  Record.
   * @param vmap  Variable mapping: Variables in 'to' --> Variables in 'from'.
   */
  void fill_record_with_record(finite_record& to, const finite_record& from,
                               const finite_var_map& vmap);

  /**
   * Fills the data in a record with data from another record,
   * using the given variable mapping.
   * @param to    (Return value) Record.
   * @param from  Record.
   * @param vmap  Variable mapping: Variables in 'to' --> Variables in 'from'.
   */
  void fill_record_with_record(vector_record& to, const vector_record& from,
                               const vector_var_map& vmap);

} // namespace sill

#endif // SILL_RECORD_CONVERSIONS_HPP

