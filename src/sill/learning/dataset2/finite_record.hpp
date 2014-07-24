#ifndef SILL_FINITE_RECORD2_HPP
#define SILL_FINITE_RECORD2_HPP

#include <iostream>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  struct finite_record2 {
    std::vector<size_t> values;
    double weight;

    typedef double weight_type;

    finite_record2() 
      : weight(0.0) { }

    explicit finite_record2(size_t n)
      : values(n), weight(0.0) { }
  };

  inline std::ostream& operator<<(std::ostream& out, const finite_record2& r) {
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
