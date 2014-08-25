#ifndef SILL_FINITE_RECORD2_HPP
#define SILL_FINITE_RECORD2_HPP

#include <iostream>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  struct finite_record {
    std::vector<size_t> values;
    double weight;

    // helper types
    typedef size_t elem_type;
    typedef double weight_type;

    finite_record() 
      : weight(0.0) { }

    explicit finite_record(size_t n, double weight = 0.0)
      : values(n), weight(weight) { }

    finite_record(const std::vector<size_t>& values, double weight)
      : values(values), weight(weight) { }

    void resize(size_t n) {
      values.resize(n);
    }

    bool operator==(const finite_record& other) const {
      return values == other.values && weight == other.weight;
    }
  };

  inline std::ostream& operator<<(std::ostream& out, const finite_record& r) {
    foreach(size_t x, r.values) {
      // there is presently no way to detect special "undefined" value here
      out << x << " ";
    }
    out << ": " << r.weight;
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
