#ifndef SILL_VARIABLES_HPP
#define SILL_VARIABLES_HPP

#include <sill/base/variable_type_group.hpp>

namespace sill {

  // Domain comparisons
  //============================================================================

  template <typename D1, typename D2>
  bool includes(const D1& a, const D2& b);

  template <>
  bool includes<domain, finite_domain>(const domain& a, const finite_domain& b);

  template <>
  bool includes<domain, vector_domain>(const domain& a, const vector_domain& b);

  template <typename D1, typename D2>
  D1 set_difference(const D1& a, const D2& b);

  template <>
  domain
  set_difference<domain, finite_domain>
  (const domain& a, const finite_domain& b);

  template <>
  domain
  set_difference<domain, vector_domain>
  (const domain& a, const vector_domain& b);

  bool set_disjoint(const domain& a, const finite_domain& b);

  bool set_disjoint(const domain& a, const vector_domain& b);

  bool set_disjoint(const finite_domain& a, const domain& b);

  bool set_disjoint(const vector_domain& a, const domain& b);

  size_t intersection_size(const domain& a, const finite_domain& b);

  size_t intersection_size(const domain& a, const vector_domain& b);

  size_t intersection_size(const finite_domain& a, const domain& b);

  size_t intersection_size(const vector_domain& a, const domain& b);

  // Domain type conversions
  //============================================================================

  //! Convert from one domain type to another.
  //! Assert false if variables are of incompatible types.
  template <typename FromVarType, typename ToVarType>
  void convert_domain
  (const typename variable_type_group<FromVarType>::domain_type& from,
   typename variable_type_group<ToVarType>::domain_type& to);

  template <>
  void convert_domain<variable, finite_variable>
  (const domain& from, finite_domain& to);

  template <>
  void convert_domain<variable, vector_variable>
  (const domain& from, vector_domain& to);

  // Variable vector conversions
  //============================================================================

  //! Extract the finite variables from the given variables.
  finite_var_vector extract_finite_var_vector(const var_vector& vars);

  //! Extract the vector variables from the given variables.
  vector_var_vector extract_vector_var_vector(const var_vector& vars);

  // Vector variable helpers
  //============================================================================

  /**
   * Compute indices for variables in vvec, based on reference set vset.
   * These indices are based on concatenating sizes of the vector variables
   * in vvec.
   * @param in_vset_indices
   *         (Return value) Indices for variables in vvec and in vset.
   * @param notin_vset_indices
   *         (Return value) Indices for variables in vvec but not in vset.
   */
  void
  vector_indices_relative_to_set(const vector_var_vector& vvec,
                                 const vector_domain& vset,
                                 ivec& in_vset_indices,
                                 ivec& notin_vset_indices);

}; // namespace sill

#endif // #ifndef SILL_VARIABLES_HPP
