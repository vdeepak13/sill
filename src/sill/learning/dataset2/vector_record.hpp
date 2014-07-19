#ifndef SILL_VECTOR_RECORD_HPP
#define SILL_VECTOR_RECORD_HPP

#include <iostream>

#include <armadillo>

#include <boost/math/special_functions/fpclassify.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T>
  struct vector_record {
    arma::Col<T> values;
    T weight;
    typedef T weight_type;
    vector_record() : weight(0.0) { }
    vector_record(size_t n) : values(n, arma::fill::zeros), weight(0.0) { }
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& out, const new_vector_record<T>& r) {
    foreach(T x, r.values) {
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
