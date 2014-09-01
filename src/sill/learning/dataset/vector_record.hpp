#ifndef SILL_VECTOR_RECORD_HPP
#define SILL_VECTOR_RECORD_HPP

#include <armadillo>
#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T = double>
  struct vector_record {
    arma::Col<T> values;
    T weight;

    // helper types
    typedef T elem_type;
    typedef T weight_type;

    vector_record()
      : weight(0.0) { }

    explicit vector_record(size_t n, T weight = 0.0)
      : values(n, arma::fill::zeros), weight(weight) { }

    vector_record(const arma::Col<T>& values, T weight)
      : values(values), weight(weight) { }

    void resize(size_t n) {
      values = arma::zeros<arma::Col<T> >(n);
    }

    bool operator==(const vector_record& other) const {
      return values.size() == other.values.size()
        && all(values == other.values)
        && weight == other.weight;
    }
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& out, const vector_record<T>& r) {
    foreach(T x, r.values) {
      out << x << " "; // nan is the special "undefined" value
    }
    out << ": " << r.weight;
    return out;
  }
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
