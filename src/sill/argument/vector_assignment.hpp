#ifndef SILL_VECTOR_ASSIGNMENT_HPP
#define SILL_VECTOR_ASSIGNMENT_HPP

#include <sill/argument/domain.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/math/eigen/dynamic.hpp>

#include <stdexcept>
#include <unordered_map>

namespace sill {

  //! \addtogroup argument_type
  //! @{

  /**
   * A type that represents an assignment to vector variables.
   * Each vector variable is mapped to an Eigen vector.
   */
  template <typename T>
  using vector_assignment =
    std::unordered_map<vector_variable*, dynamic_vector<T> >;
  
  /**
   * Returns the aggregate size of all vector variables in the assignment.
   * \relates vector_assignment
   */
  template <typename T>
  size_t vector_size(const vector_assignment<T>& a) {
    size_t size = 0;
    for (const auto& p : a) {
      size += p.first->size();
    }
    return size;
  }

  /**
   * Returns the concatenation of (a subset of) vectors in an assignment
   * in the order specified by the given domain.
   */
  template <typename T>
  dynamic_vector<T>
  extract(const vector_assignment<T>& a,
          const domain<vector_variable*>& dom) {
    dynamic_vector<T> result(vector_size(dom));
    size_t i = 0;
    for (vector_variable* v : dom) {
      result.segment(i, v->size()) = a.at(v);
      i += v->size();
    }
    return result;
  }

  //! @}

} // namespace sill

#endif
