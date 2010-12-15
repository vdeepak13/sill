
#include <prl/learning/dataset/datasource_info_type.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  datasource_info_type::datasource_info_type
  (const finite_var_vector& finite_seq,
   const vector_var_vector& vector_seq,
   const std::vector<variable::variable_typenames>& var_type_order,
   const finite_var_vector& finite_class_vars,
   const vector_var_vector& vector_class_vars)
    : finite_seq(finite_seq), vector_seq(vector_seq),
      var_type_order(var_type_order),
      finite_class_vars(finite_class_vars),
      vector_class_vars(vector_class_vars) {
  }

  datasource_info_type::
  datasource_info_type(const forward_range<variable*>& var_seq) {
    foreach(variable* var, var_seq) {
      switch (var->get_variable_type()) {
      case variable::FINITE_VARIABLE:
        finite_seq.push_back((finite_variable*)var);
        var_type_order.push_back(variable::FINITE_VARIABLE);
        break;
      case variable::VECTOR_VARIABLE:
        vector_seq.push_back((vector_variable*)var);
        var_type_order.push_back(variable::VECTOR_VARIABLE);
        break;
      default:
        assert(false);
      }
    }
  }

  /*
  datasource_info_type::
  datasource_info_type(const forward_range<finite_variable*>& var_seq)
    : finite_seq(var_seq.begin(), var_seq.end()),
      var_type_order(finite_seq.size(), variable::FINITE_VARIABLE) {
  }

  datasource_info_type::
  datasource_info_type(const forward_range<vector_variable*>& var_seq)
    : vector_seq(var_seq.begin(), var_seq.end()),
      var_type_order(vector_seq.size(), variable::VECTOR_VARIABLE) {
  }
  */

}; // namespace prl

#include <prl/macros_undef.hpp>
