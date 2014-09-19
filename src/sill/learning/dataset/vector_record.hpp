#ifndef SILL_VECTOR_RECORD_HPP
#define SILL_VECTOR_RECORD_HPP

#include <sill/base/vector_variable.hpp>

#include <armadillo>
#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T = double>
  struct vector_record {
    vector_var_vector variables;
    arma::Col<T> values;
    T weight;

    // helper types
    typedef T elem_type;
    typedef T weight_type;

    vector_record()
      : weight(0.0) { }

    explicit vector_record(const vector_var_vector& vars, T weight = 1.0)
      : variables(vars),
        values(vector_size(vars), arma::fill::zeros),
        weight(weight) { }

    vector_record(const vector_var_vector& vars,
                  const arma::Col<T>& values,
                  T weight)
      : variables(vars),
        values(values),
        weight(weight) { }

    bool operator==(const vector_record& other) const {
      return values.size() == other.values.size()
        && all(values == other.values)
        && weight == other.weight;
    }

    void extract(vector_assignment& a) const {
      size_t col = 0;
      foreach (vector_variable* v, variables) {
        arma::Col<T>& value = a[v];
        value.set_size(v->size());
        for (size_t i = 0; i < v->size(); ) {
          value[i++] = values[col++];
        }
      }
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
