
#ifndef SILL_RECORD_CONVERSIONS_HPP
#define SILL_RECORD_CONVERSIONS_HPP

#include <sill/base/assignment.hpp>
#include <sill/learning/dataset/record.hpp>

namespace sill {

  /**
   * Converts the given finite assignment into finite record data.
   * @param fa          Source data.
   * @param findata     Target data.
   * @param finite_seq  Finite variable ordering used for findata.
   */
  void
  finite_assignment2record(const finite_assignment& fa,
                           std::vector<size_t>& findata,
                           const finite_var_vector& finite_seq);

  /**
   * Converts the given vector assignment into vector record data.
   * @param fa          Source data.
   * @param vecdata     Target data.
   * @param vector_seq  Vector variable ordering used for vecdata.
   */
  void
  vector_assignment2record(const vector_assignment& va,
                           vec& vecdata,
                           const vector_var_vector& vector_seq);

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

