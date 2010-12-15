
#ifndef PRL_VARIABLE_TYPE_UNION_HPP
#define PRL_VARIABLE_TYPE_UNION_HPP

#include <prl/base/finite_variable.hpp>
#include <prl/base/vector_variable.hpp>

namespace prl {

  /**
   * Struct which, given two variable types as template parameters, specifies
   * the variable type which is their union:
   *  - union_type
   */
  template <typename V1, typename V2>
  struct variable_type_union {
  };

  template <> struct variable_type_union <finite_variable, finite_variable> {
    typedef finite_variable union_type;
  };

  template <> struct variable_type_union <vector_variable, vector_variable> {
    typedef vector_variable union_type;
  };

  template <> struct variable_type_union <finite_variable, vector_variable> {
    typedef variable union_type;
  };

  template <> struct variable_type_union <vector_variable, finite_variable> {
    typedef variable union_type;
  };

  template <> struct variable_type_union <variable, finite_variable> {
    typedef variable union_type;
  };

  template <> struct variable_type_union <finite_variable, variable> {
    typedef variable union_type;
  };

  template <> struct variable_type_union <variable, vector_variable> {
    typedef variable union_type;
  };

  template <> struct variable_type_union <vector_variable, variable> {
    typedef variable union_type;
  };

  template <> struct variable_type_union <variable, variable> {
    typedef variable union_type;
  };

} // namespace prl

#endif // #ifndef PRL_VARIABLE_TYPE_UNION_HPP
