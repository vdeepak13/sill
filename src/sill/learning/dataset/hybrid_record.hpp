#ifndef SILL_HYBRID_RECORD_HPP
#define SILL_HYBRID_RECORD_HPP

#include <sill/base/hybrid_values.hpp>

#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T = double>
  struct hybrid_record {
    hybrid_values<T> values;
    T weight;
    
    hybrid_record()
      : weight(0.0) { }

    hybrid_record(size_t nfinite, size_t nvector, T weight = 0.0)
      : values(nfinite, nvector), weight(weight) { }

    hybrid_record(const std::vector<size_t>& finite,
                  const arma::Col<T>& vector,
                  T weight)
      : values(finite, vector), weight(weight) { }

    void resize(size_t nfinite, size_t nvector) {
      values.resize(nfinite, nvector);
    }

    size_t size() const {
      return values.finite.size() + values.vector.size();
    }

    bool operator==(const hybrid_record& other) const {
      return values == other.values && weight == other.weight;
    }
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& out, const hybrid_record<T>& r) {
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
