#ifndef SILL_VARIABLE_UTILS_HPP
#define SILL_VARIABLE_UTILS_HPP

#include <sill/base/variable.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/range/converted.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Returns the union of a finite and a vector domain.
   * \relates variable
   */
  inline domain set_union(const finite_domain& s,
                          const vector_domain& t) {
    domain result;
    sill::set_union(make_converted<variable*>(s),
                    make_converted<variable*>(t),
                    std::inserter(result, result.begin()));
    return result;
  }

  /**
   * Returns the union of a finite and a vector domain.
   * \relates variable
   */
  inline domain set_union(const vector_domain& s,
                          const finite_domain& t) {
    domain result;
    sill::set_union(make_converted<variable*>(s),
                    make_converted<variable*>(t),
                    std::inserter(result, result.begin()));
    return result;
  }

  inline domain set_union(const domain& s, const finite_domain& t) {
    domain result;
    sill::set_union(s, make_converted<variable*>(t),
                    std::inserter(result, result.begin()));
    return result;
  }

  inline domain set_union(const domain& s, const vector_domain& t) {
    domain result;
    sill::set_union(s, make_converted<variable*>(t),
                    std::inserter(result, result.begin()));
    return result;
  }

  /**
   * Returns the concatentaion of finite and vector variables.
   * \relates variable
   */
  inline var_vector concat(const finite_var_vector& u,
                           const vector_var_vector& v) {
    var_vector result;
    result.insert(result.end(), u.begin(), u.end());
    result.insert(result.end(), v.begin(), v.end());
    return result;
  }

  /**
   * Returns the concatentaion of vector and finite variables.
   * \relates variable
   */
  inline var_vector concat(const vector_var_vector& u,
                           const finite_var_vector& v) {
    var_vector result;
    result.insert(result.end(), u.begin(), u.end());
    result.insert(result.end(), v.begin(), v.end());
    return result;
  }

  /**
   * Splits the vector of variables into finite and vectors,
   * maintaining the ordering.
   */
  inline void split(const var_vector& vars,
                    finite_var_vector& finite_vars,
                    vector_var_vector& vector_vars) {
    foreach(variable* v, vars) {
      switch (v->type()) {
      case variable::FINITE_VARIABLE:
        finite_vars.push_back(dynamic_cast<finite_variable*>(v));
        break;
      case variable::VECTOR_VARIABLE:
        vector_vars.push_back(dynamic_cast<vector_variable*>(v));
        break;
      }
    }
  }

  /**
   * Returns all variables from the vector that are also present in
   * the given associative container (set or map).
   */
  template <typename V, typename Container>
  std::vector<V*>
  intersect(const std::vector<V*>& vec, const Container& container) {
    std::vector<V*> result;
    foreach (V* v, vec) {
      if (container.count(v)) {
        result.push_back(v);
      }
    }
    return result;
  }

  /**
   * Returns all variables from the vector that are not present in
   * the given associative container (set or map).
   */
  template <typename V, typename Container>
  std::vector<V*>
  difference(const std::vector<V*>& vec, const Container& container) {
    std::vector<V*> result;
    foreach (V* v, vec) {
      if (!container.count(v)) {
        result.push_back(v);
      }
    }
    return result;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
