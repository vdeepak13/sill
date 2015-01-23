#ifndef SILL_HYBRID_RECORD_HPP
#define SILL_HYBRID_RECORD_HPP

#include <sill/base/assignment.hpp>
#include <sill/datastructure/hybrid_index.hpp>
#include <sill/base/variable_utils.hpp>

#include <iostream>

#include <boost/math/special_functions/fpclassify.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents one data point over finite and vector variables.
   * Models the Record concept.
   *
   * \tparam T the storage type of the values stored in this record
   */
  template <typename T = double>
  struct hybrid_record {

    //! The sequence of finite variables stored in this record
    finite_var_vector finite_vars;

    //! The sequence of vector variables stored in this record
    vector_var_vector vector_vars;

    //! The values (finite and vector) stored in this record
    hybrid_index<T> values;

    //! The weight of the data point
    T weight;
    
    // Record concept types
    typedef var_vector       var_vector_type;
    typedef assignment       assignment_type;
    typedef hybrid_index<T> values_type;
    typedef T                weight_type;

    /**
     * Consturcts an empty record with no values and zero weight.
     */
    hybrid_record()
      : weight(0.0) { }

    /**
     * Constructs a record with given variables and weight,
     * setting all the values to 0.
     */
    explicit hybrid_record(const var_vector& vars, T weight = 0.0)
      : weight(weight) {
      split(vars, finite_vars, vector_vars);
      values.resize(finite_vars.size(), vector_size(vector_vars));
    }

    /**
     * Constructs a record with the given variables and weight.
     */
    hybrid_record(const finite_var_vector& finite_vars,
                  const vector_var_vector& vector_vars,
                  T weight = 0.0)
      : finite_vars(finite_vars),
        vector_vars(vector_vars),
        values(finite_vars.size(), vector_size(vector_vars)),
        weight(weight) { }

    /**
     * Returns the number of variables (finite or vector).
     */
    size_t size() const {
      return values.finite.size() + values.vector.size();
    }

    /**
     * Returns true if two records are equal.
     */
    bool operator==(const hybrid_record& other) const {
      return values == other.values && weight == other.weight;
    }

    /**
     * Extracts the values from this record into an assignment.
     * The assignment is not cleared; instead, any existing values
     * are overwritten, adn missing values are erased.
     */
    void extract(assignment& a) const {
      assert(finite_vars.size() == values.finite.size());
      for (size_t i = 0; i < finite_vars.size(); ++i) {
        if (values.finite[i] == size_t(-1)) {
          a.erase(finite_vars[i]);
        } else {
          a[finite_vars[i]] = values.finite[i];
        }
      }
      size_t col = 0;
      foreach (vector_variable* v, vector_vars) {
        if (boost::math::isnan(values.vector[col])) {
          a.erase(v);
          col += v->size();
        } else {
          arma::Col<T>& value = a[v];
          value.set_size(v->size());
          for (size_t i = 0; i < v->size(); ) {
            value[i++] = values.vector[col++];
          }
        }
      }
    }

    /**
     * Returns the number of missing values in this record.
     */
    size_t count_missing() const {
      size_t count = 
        std::count(values.finite.begin(), values.finite.end(), size_t(-1));
      size_t col = 0;
      foreach (vector_variable* v, vector_vars) {
        count += boost::math::isnan(values.vector[col]);
        col += v->size();
      }
      return count;
    }

    /**
     * Returns the number of missing values in this record among the
     * specified variables. The specified variables must form
     * a subsequence of the variables stored in this record.
     */
    size_t count_missing(const var_vector& subseq) const {
      size_t count = 0;
      size_t fi = 0;
      size_t vi = 0;
      size_t col = 0;
      foreach (variable* v, subseq) {
        switch (v->type()) {
        case variable::FINITE_VARIABLE:
          while (finite_vars[fi] != v) {
            ++fi;
            assert(fi < finite_vars.size());
          }
          count += values.finite[fi] == size_t(-1);
          break;
        case variable::VECTOR_VARIABLE:
          while (vector_vars[vi] != v) {
            col += vector_vars[vi]->size();
            ++vi;
            assert(vi < vector_vars.size());
          }
          count += boost::math::isnan(values.vector[col]);
          break;
        }
      }
      return count;
    }

  }; // class hybrid_record

  template <typename T>
  std::ostream& operator<<(std::ostream& out, const hybrid_record<T>& r) {
    foreach(size_t x, r.values.finite) {
      if (x == size_t(-1)) {
        out << "NA ";
      } else {
        out << x << " ";
      }
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
