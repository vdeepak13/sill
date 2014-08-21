#ifndef SILL_VARIABLE_TYPE_GROUP_HPP
#define SILL_VARIABLE_TYPE_GROUP_HPP

#include <sill/base/assignment.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>

namespace sill {

  // Forward declarations
  class finite_record;
  template <typename LA> class vector_record;
  template <typename LA> class record;

  /**
   * Struct which, given a variable type as a template parameter, specifies:
   *  - domain_type
   *  - var_vector_type
   *  - assignment_type
   *  - record_type
   *
   * @tparam V   Variable type.
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename V, typename LA = dense_linear_algebra<> >
  struct variable_type_group {
  };

  template <typename LA>
  struct variable_type_group <finite_variable, LA> {

    typedef finite_domain domain_type;

    typedef finite_var_vector var_vector_type;

    typedef finite_assignment assignment_type;

    typedef finite_record record_type;

  };

  template <typename LA>
  struct variable_type_group <vector_variable, LA> {

    typedef vector_domain domain_type;

    typedef vector_var_vector var_vector_type;

    typedef vector_assignment assignment_type;

    typedef vector_record<LA> record_type;

  };

  template <typename LA>
  struct variable_type_group <variable, LA> {

    typedef domain domain_type;

    typedef var_vector var_vector_type;

    typedef assignment assignment_type;

    typedef record<LA> record_type;

  };

} // namespace sill

#include <sill/learning/dataset/record.hpp>

#endif // #ifndef SILL_VARIABLE_TYPE_GROUP_HPP
