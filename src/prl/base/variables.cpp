
#include <sill/base/variables.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Domain comparisons
  //============================================================================

  template <>
  bool
  includes<domain, finite_domain>(const domain& a, const finite_domain& b) {
    return std::includes(a.begin(), a.end(), b.begin(), b.end());
  }

  template <>
  bool
  includes<domain, vector_domain>(const domain& a, const vector_domain& b) {
    return std::includes(a.begin(), a.end(), b.begin(), b.end());
  }

  template <>
  domain
  set_difference<domain, finite_domain>
  (const domain& a, const finite_domain& b) {
    domain output;
    std::set_difference(a.begin(), a.end(), 
                        b.begin(), b.end(),
                        std::inserter(output, output.begin()));
    return output;
  }

  template <>
  domain
  set_difference<domain, vector_domain>
  (const domain& a, const vector_domain& b) {
    domain output;
    std::set_difference(a.begin(), a.end(), 
                        b.begin(), b.end(),
                        std::inserter(output, output.begin()));
    return output;
  }

  bool set_disjoint(const domain& a, const finite_domain& b) {
    return (intersection_size(a,b) == 0);
  }

  bool set_disjoint(const domain& a, const vector_domain& b) {
    return (intersection_size(a,b) == 0);
  }

  bool set_disjoint(const finite_domain& a, const domain& b) {
    return (intersection_size(a,b) == 0);
  }

  bool set_disjoint(const vector_domain& a, const domain& b) {
    return (intersection_size(a,b) == 0);
  }

  size_t intersection_size(const domain& a, const finite_domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  size_t intersection_size(const domain& a, const vector_domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  size_t intersection_size(const finite_domain& a, const domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  size_t intersection_size(const vector_domain& a, const domain& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  // Domain type conversions
  //============================================================================

  template <>
  void convert_domain<variable, finite_variable>
  (const domain& from, finite_domain& to) {
    to.clear();
    foreach(variable* v, from) {
      if (v->get_variable_type() == variable::FINITE_VARIABLE)
        to.insert((finite_variable*)v);
      else
        assert(false);
    }
  }

  template <>
  void convert_domain<variable, vector_variable>
  (const domain& from, vector_domain& to) {
    to.clear();
    foreach(variable* v, from) {
      if (v->get_variable_type() == variable::VECTOR_VARIABLE)
        to.insert((vector_variable*)v);
      else
        assert(false);
    }
  }

  // Variable vector conversions
  //============================================================================

  //! Extract the finite variables from the given variables.
  finite_var_vector extract_finite_var_vector(const var_vector& vars) {
    finite_var_vector fvars;
    foreach(variable* v, vars) {
      if (v->get_variable_type() == variable::FINITE_VARIABLE)
        fvars.push_back((finite_variable*)v);
    }
    return fvars;
  }

  //! Extract the vector variables from the given variables.
  vector_var_vector extract_vector_var_vector(const var_vector& vars) {
    vector_var_vector vvars;
    foreach(variable* v, vars) {
      if (v->get_variable_type() == variable::VECTOR_VARIABLE)
        vvars.push_back((vector_variable*)v);
    }
    return vvars;
  }

}; // namespace sill

#include <sill/macros_undef.hpp>
