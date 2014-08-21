#ifndef SILL_RECORD_CONVERSIONS_HPP
#define SILL_RECORD_CONVERSIONS_HPP

#include <sill/base/assignment.hpp>
#include <sill/math/linear_algebra/sparse_linear_algebra.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Adds the given vector x matching vector variables X to the assignment.
   */
  template <typename VecType>
  void add_vector2vector_assignment(const vector_var_vector& X,
                                    const VecType& x,
                                    vector_assignment& va);

  /**
   * Converts the given finite assignment into finite record data.
   * @param fa          Source data.
   * @param finite_seq  Finite variable ordering used for findata.
   * @param findata     Target data.
   */
  void
  finite_assignment2vector(const finite_assignment& fa,
                           const finite_var_vector& finite_seq,
                           std::vector<size_t>& findata);

  /**
   * Converts the given vector assignment into vector record data.
   * @param fa          Source data.
   * @param vector_seq  Vector variable ordering used for vecdata.
   * @param vecdata     Target data.
   */
  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           vec& vecdata);

  /**
   * Converts the given vector assignment into vector record data.
   * @param fa          Source data.
   * @param vector_seq  Vector variable ordering used for vecdata.
   * @param vecdata     Target data.
   */
  template <typename T, typename Index>
  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           sparse_vector<T,Index>& vecdata);

  //============================================================================
  // Implementations of above functions
  //============================================================================

  template <typename VecType>
  void add_vector2vector_assignment(const vector_var_vector& X,
                                    const VecType& x,
                                    vector_assignment& va) {
    size_t i = 0; // index into x
    vec val;
    foreach(vector_variable* v, X) {
      val.set_size(v->size());
      if (i + v->size() > x.size()) {
        throw std::invalid_argument
          (std::string("add_vector2vector_assignment(X,x,va)") +
           " given X,x of non-matching dimensionalities (|X| > |x|).");
      }
      for (size_t j = 0; j < v->size(); ++j)
        val[j] = x[i++];
      va[v] = val;
    }
    if (i != x.size()) {
        throw std::invalid_argument
          (std::string("add_vector2vector_assignment(X,x,va)") +
           " given X,x of non-matching dimensionalities (|X| < |x|).");
    }
  }

  template <typename T, typename Index>
  void
  vector_assignment2vector(const vector_assignment& va,
                           const vector_var_vector& vector_seq,
                           sparse_vector<T,Index>& vecdata) {
    // Try to preserve sparsity.
    size_t n = vector_size(vector_seq);
    size_t i = 0;
    std::vector<Index> inds;
    std::vector<T> vals;
    vector_assignment::const_iterator va_end = va.end();
    foreach(vector_variable* v, vector_seq) {
      vector_assignment::const_iterator it(va.find(v));
      if (it == va_end) {
        throw std::runtime_error
          (std::string("vector_assignment2vector(va,vector_seq,vecdata)") +
           " given vector_seq with variables not appearing in given" +
           " assignment.");
      }
      const vec& tmpvec = it->second;
      for (size_t j(0); j < v->size(); j++) {
        if (tmpvec[j] != 0) {
          inds.push_back(i + j);
          vals.push_back(tmpvec[i + j]);
        }
      }
      i += v->size();
    }
    vecdata.reset(n, inds, vals);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_RECORD_CONVERSIONS_HPP

