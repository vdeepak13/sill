#ifndef SILL_RECORD_HPP
#define SILL_RECORD_HPP

#include <iostream>

#include <sill/base/hybrid_values.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T>
  struct record {
    hybrid_values<T> values;
    const T weight;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& out, const record<T>& r) {
    foreach(size_t x, r.values.finite) {
      // there is presently no way to detect special "undefined" value here
      out << x << " ";
    }
    out << ": ";
    foreach(T x, r.values.vector) {
      out << x << " "; // nan is the special "undefined" value
    }
    out << ": " << r.weight;
    return out;
  }
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
