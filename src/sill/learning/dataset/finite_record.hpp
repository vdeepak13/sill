#ifndef SILL_FINITE_RECORD_HPP
#define SILL_FINITE_RECORD_HPP

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>

#include <iostream>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  struct finite_record {
    finite_var_vector variables;
    std::vector<size_t> values;
    double weight;

    // helper types
    typedef size_t elem_type;
    typedef double weight_type;

    finite_record() 
      : weight(0.0) { }

    explicit finite_record(const finite_var_vector& vars, double weight = 1.0)
      : variables(vars), values(vars.size()), weight(weight) { }

    finite_record(const finite_var_vector& vars,
                  const std::vector<size_t>& values,
                  double weight)
      : variables(vars), values(values), weight(weight) {
      assert(vars.size() == values.size());
    }

    bool operator==(const finite_record& other) const {
      return values == other.values && weight == other.weight;
    }
    
    void extract(finite_assignment& a) const {
      assert(variables.size() == values.size());
      for (size_t i = 0; i < variables.size(); ++i) {
        a[variables[i]] = values[i];
      }
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
