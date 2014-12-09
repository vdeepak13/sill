#ifndef SILL_VECTOR_RECORD_HPP
#define SILL_VECTOR_RECORD_HPP

#include <sill/base/vector_variable.hpp>
#include <sill/base/vector_assignment.hpp>

#include <armadillo>
#include <iostream>

#include <boost/math/special_functions/fpclassify.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents one data point over vector variables. 
   * Models the Record concept.
   *
   * \tparam T the storage type of the values stored in this record
   */
  template <typename T = double>
  struct vector_record {

    //! The sequence of variables stored in this record
    vector_var_vector variables;
    
    //! The values stored in this record
    arma::Col<T> values;

    //! The weight of the data point
    T weight;

    // Record concept types
    typedef vector_var_vector var_vector_type;
    typedef vector_assignment assignment_type;
    typedef arma::Col<T>      values_type;
    typedef T                 weight_type;

    // helper types for raw_record_iterators
    typedef T elem_type;

    /**
     * Constructs an empty record with no values and zero weight.
     */
    vector_record()
      : weight(0.0) { }

    /**
     * Constructs a record with the given variables and weight,
     * setting all the values to 0.
     */
    explicit vector_record(const vector_var_vector& vars, T weight = 1.0)
      : variables(vars),
        values(vector_size(vars), arma::fill::zeros),
        weight(weight) { }

    /**
     * Constructs a record with the given variables, values, and weight.
     */
    vector_record(const vector_var_vector& vars,
                  const arma::Col<T>& values,
                  T weight)
      : variables(vars),
        values(values),
        weight(weight) { }

    /**
     * Returns true if two record are equal.
     */
    bool operator==(const vector_record& other) const {
      return values.size() == other.values.size()
        && all(values == other.values)
        && weight == other.weight;
    }

    /**
     * Extracts the values from this record into an assignment.
     * The assignment is not cleared; instead, any existing values
     * are overritten, and missing values are erased.
     */
    void extract(vector_assignment& a) const {
      size_t col = 0;
      foreach (vector_variable* v, variables) {
        if (boost::math::isnan(values[col])) {
          a.erase(v);
          col += v->size();
        } else {
          arma::Col<T>& value = a[v];
          value.set_size(v->size());
          for (size_t i = 0; i < v->size(); ) {
            value[i++] = values[col++];
          }
        }
      }
    }
    
    /**
     * Returns the number of missing values in this record.
     */
    size_t count_missing() const {
      size_t count = 0;
      size_t col = 0;
      foreach (vector_variable* v, variables) {
        count += boost::math::isnan(values[col]);
        col += v->size();
      }
      return count;
    }

    /**
     * Returns the number of missing values in this record among the
     * specified variables. The specified variables must form
     * a subsequence of the variables stored in this record.
     */
    size_t count_missing(const vector_var_vector& subseq) const {
      size_t count = 0;
      size_t col = 0;
      size_t j = 0;
      for (size_t i = 0; i < variables.size() && j < subseq.size(); ++i) {
        if (variables[i] == subseq[j]) {
          count += boost::math::isnan(values[col]);
          ++j;
        }
        col += variables[i]->size();
      }
      assert(j == subseq.size());
      return count;
    }

  }; // class vector_record

  /**
   * Prints a vector record to an output stream.
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const vector_record<T>& r) {
    foreach(T x, r.values) {
      out << x << " "; // nan is the missing value
    }
    out << ": " << r.weight;
    return out;
  }
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
