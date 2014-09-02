#ifndef SILL_HYBRID_RECORD_HPP
#define SILL_HYBRID_RECORD_HPP

#include <sill/base/hybrid_values.hpp>

#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T = double>
  struct hybrid_record {
    finite_var_vector finite_vars;
    vector_var_vector vector_vars;
    hybrid_values<T> values;
    T weight;
    
    hybrid_record()
      : weight(0.0) { }

    explicit hybrid_record(const var_vector& vars, T weight = 0.0)
      : weight(weight) {
      split(vars, finite_vars, vector_vars);
      values.resize(finite_vars.size(), vector_size(vector_vars));
    }

    hybrid_record(const finite_var_vector& finite_vars,
                  const vector_var_vector& vector_vars,
                  T weight = 0.0)
      : finite_vars(finite_vars),
        vector_vars(vector_vars),
        values(finite_vars.size(), vector_size(vector_vars)),
        weight(weight) { }

    size_t size() const {
      return values.finite.size() + values.vector.size();
    }

    bool operator==(const hybrid_record& other) const {
      return values == other.values && weight == other.weight;
    }

    void extract(assignment& a) const {
      assert(finite_vars.size() == values.finite.size());
      for (size_t i = 0; i < finite_vars.size(); ++i) {
        a[finite_vars[i]] = values.finite[i];
      }
      size_t col = 0;
      foreach (vector_variable* v, vector_vars) {
        arma::Col<T>& value = a[v];
        value.set_size(v->size());
        for (size_t i = 0; i < v->size(); ) {
          value[i++] = values.vector[col++];
        }
      }
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
