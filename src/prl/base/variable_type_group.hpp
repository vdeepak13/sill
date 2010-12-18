
#ifndef SILL_VARIABLE_TYPE_GROUP_HPP
#define SILL_VARIABLE_TYPE_GROUP_HPP

#include <sill/base/assignment.hpp>

namespace sill {

  // Forward declarations
  class finite_record;
  class vector_record;
  class record;

  /**
   * Struct which, given a variable type as a template parameter, specifies:
   *  - domain_type
   *  - var_vector_type
   *  - assignment_type
   *  - record_type
   */
  template <typename V>
  struct variable_type_group {
  };

  template <>
  struct variable_type_group <finite_variable> {

    typedef finite_domain domain_type;

    typedef finite_var_vector var_vector_type;

    typedef finite_assignment assignment_type;

    typedef finite_record record_type;

    typedef finite_var_map var_map_type;

  };

  template <>
  struct variable_type_group <vector_variable> {

    typedef vector_domain domain_type;

    typedef vector_var_vector var_vector_type;

    typedef vector_assignment assignment_type;

    typedef vector_record record_type;

    typedef vector_var_map var_map_type;

  };

  template <>
  struct variable_type_group <variable> {

    typedef domain domain_type;

    typedef var_vector var_vector_type;

    typedef assignment assignment_type;

    typedef record record_type;

    typedef var_map var_map_type;

  };

} // namespace sill

#include <sill/learning/dataset/record.hpp>

#endif // #ifndef SILL_VARIABLE_TYPE_GROUP_HPP
