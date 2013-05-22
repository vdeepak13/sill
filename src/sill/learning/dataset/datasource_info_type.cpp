
#include <sill/learning/dataset/datasource_info_type.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  datasource_info_type::datasource_info_type() { }

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
    // TO DO: CHECK CONSISTENCY
  }

  datasource_info_type::datasource_info_type
  (const forward_range<finite_variable*>& finite_seq,
   const forward_range<vector_variable*>& vector_seq,
   const std::vector<variable::variable_typenames>& var_type_order)
    : finite_seq(finite_seq.begin(), finite_seq.end()),
      vector_seq(vector_seq.begin(), vector_seq.end()),
      var_type_order(var_type_order) {
    // TO DO: CHECK CONSISTENCY
  }

  datasource_info_type::datasource_info_type
  (const forward_range<finite_variable*>& finite_seq_,
   const forward_range<vector_variable*>& vector_seq_)
    : finite_seq(finite_seq_.begin(), finite_seq_.end()),
      vector_seq(vector_seq_.begin(), vector_seq_.end()) {
    var_type_order.reserve(finite_seq.size() + vector_seq.size());
    for (size_t i = 0; i < finite_seq.size(); ++i)
      var_type_order.push_back(variable::FINITE_VARIABLE);
    for (size_t i = 0; i < vector_seq.size(); ++i)
      var_type_order.push_back(variable::VECTOR_VARIABLE);
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

  datasource_info_type::
  datasource_info_type(const finite_var_vector& var_seq)
    : finite_seq(var_seq),
      var_type_order(finite_seq.size(), variable::FINITE_VARIABLE) {
  }

  datasource_info_type::
  datasource_info_type(const vector_var_vector& var_seq)
    : vector_seq(var_seq),
      var_type_order(vector_seq.size(), variable::VECTOR_VARIABLE) {
  }

  void datasource_info_type::save(oarchive& a) const {
    a << finite_seq << vector_seq << var_type_order
      << finite_class_vars << vector_class_vars;
  }

  void datasource_info_type::load(iarchive& a) {
    a >> finite_seq >> vector_seq >> var_type_order
      >> finite_class_vars >> vector_class_vars;
  }

  bool
  datasource_info_type::operator==(const datasource_info_type& other) const {
    if (finite_seq == other.finite_seq &&
        vector_seq == other.vector_seq &&
        var_type_order == other.var_type_order &&
        finite_class_vars == other.finite_class_vars &&
        vector_class_vars == other.vector_class_vars)
      return true;
    else
      return false;
  }

  bool
  datasource_info_type::operator!=(const datasource_info_type& other) const {
    return !operator==(other);
  }

}; // namespace sill

#include <sill/macros_undef.hpp>
