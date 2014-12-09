#ifndef SILL_FINITE_RECORD_HPP
#define SILL_FINITE_RECORD_HPP

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>

#include <iostream>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents one data point over finite variables. 
   * Models the Record concept.
   */
  struct finite_record {

    //! The sequence of variables stored in this record
    finite_var_vector variables;

    //! The values stored in this record
    std::vector<size_t> values;

    //! The weight of the data point
    double weight;

    // Record concept types
    typedef finite_var_vector   var_vector_type;
    typedef finite_assignment   assignment_type;
    typedef std::vector<size_t> values_type;
    typedef double              weight_type;

    // helper types for raw_record_iterators
    typedef size_t elem_type;

    /**
     * Constructs an empty record with no values and zero weight.
     */
    finite_record() 
      : weight(0.0) { }

    /**
     * Constructs a record with the given variables and weight,
     * setting all the values to 0.
     */
    explicit finite_record(const finite_var_vector& vars, double weight = 1.0)
      : variables(vars), values(vars.size()), weight(weight) { }

    /**
     * Constructs a record with the given variables, values, and weight.
     */
    finite_record(const finite_var_vector& vars,
                  const std::vector<size_t>& values,
                  double weight)
      : variables(vars), values(values), weight(weight) {
      assert(vars.size() == values.size());
    }

    /**
     * Returns true if two records are equal.
     */
    bool operator==(const finite_record& other) const {
      return values == other.values && weight == other.weight;
    }
    
    /**
     * Extracts the values from this record into an assignment.
     * The assignment is not cleared; instead, any existing values
     * are overritten, and missing values are erased.
     */
    void extract(finite_assignment& a) const {
      assert(variables.size() == values.size());
      for (size_t i = 0; i < variables.size(); ++i) {
        if (values[i] == size_t(-1)) {
          a.erase(variables[i]);
        } else {
          a[variables[i]] = values[i];
        }
      }
    }

    /**
     * Returns the number of missing values in this record.
     */
    size_t count_missing() const {
      return std::count(values.begin(), values.end(), size_t(-1));
    }

    /**
     * Returns the number of missing values in this record among the
     * specified variables. The specified variables must form
     * a subsequence of the variables stored in this record.
     */
    size_t count_missing(const finite_var_vector& subseq) const {
      size_t count = 0;
      size_t j = 0;
      for (size_t i = 0; i < variables.size() && j < subseq.size(); ++i) {
        if (variables[i] == subseq[j]) {
          count += (values[i] == size_t(-1));
          ++j;
        }
      }
      assert(j == subseq.size());
      return count;
    }

  }; // class finite_record

  /**
   * Prints a finite record to an output stream.
   */
  inline std::ostream& operator<<(std::ostream& out, const finite_record& r) {
    foreach(size_t x, r.values) {
      if (x == size_t(-1)) {
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
