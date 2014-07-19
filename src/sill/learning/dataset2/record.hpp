#ifndef SILL_RECORD_HPP
#define SILL_RECORD_HPP

#include <iostream>

#include <sill/base/hybrid_values.hpp>

#include <boost/math/special_functions/fpclassify.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T>
  struct record {
    hybrid_values<T> values;
    const T weight;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& out, const record<T>& r) {
    foreach(size_t x, r.index.finite) {
      if (x == size_t(-1)) {
        out << "NA ";
      } else {
        out << x << " ";
      }
    }
    out << ": ";
    foreach(T x, r.index.vector) {
      if (boost::isnan(x)) {
        out << "NA ";
      } else {
        out << x << " ";
      }
    }
    out << ": " << r.weight;
    return out;
  }
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
