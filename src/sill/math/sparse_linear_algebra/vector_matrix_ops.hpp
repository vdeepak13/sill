
#ifndef _SILL_VECTOR_MATRIX_OPS_HPP_
#define _SILL_VECTOR_MATRIX_OPS_HPP_

#include <sill/math/matrix.hpp>
#include <sill/math/sparse_linear_algebra/coo_matrix.hpp>
#include <sill/math/sparse_linear_algebra/csc_matrix.hpp>
#include <sill/math/sparse_linear_algebra/sparse_vector.hpp>
#include <sill/math/vector.hpp>

/**
 * \file vector_matrix_ops.hpp  Free functions for vectors and matrices.
 *
 * File contents:
 */

namespace sill {

  //============================================================================
  // Vector-Vector operations
  //============================================================================

  //! Dot product.
  template <typename T, typename Index>
  T dot(const dense_vector_view<T,Index>& x, const sparse_vector_i<T,Index>& y);

  //! Outer product
  template <typename T, typename Index>
  csc_matrix<T,Index> outer_product(const dense_vector_view<T,Index>& x,
                                    const sparse_vector_i<T,Index>& y);

  //============================================================================
  // Matrix-Vector operations
  //============================================================================

  //! Dense matrix  x  sparse vector --> dense vector
  template <typename T, typename Index>
  vector<T> operator*(const matrix<T>& m, const sparse_vector_i<T,Index>& v);


  //============================================================================
  // Vector-Vector operations: implementations
  //============================================================================

  //! Dot product.
  template <typename T, typename Index>
  T dot(const dense_vector_view<T,Index>& x, const sparse_vector_i<T,Index>& y){
    assert(x.size() == y.size());
    T r = 0;
    for (Index i = 0; i < y.num_non_zeros(); ++i)
      r += x[y.index(i)] * y.value(i);
    return r;
  }

  //! Outer product
  template <typename T, typename Index>
  csc_matrix<T,Index> outer_product(const dense_vector_view<T,Index>& x,
                                    const sparse_vector_i<T,Index>& y) {
    csc_matrix<T,Index> r;
    vector<Index> col_offsets(y.num_non_zeros() + 1);
    vector<Index> row_indices(x.size() * y.num_non_zeros());
    vector<T> values(row_indices.size());
    Index k = 0;
    for (Index j_ = 0; j_ < y.num_non_zeros(); ++j_) {
      Index j = y.index(j_);
      col_offsets[j] = k;
      for (Index i = 0; i < x.size(); ++i) {
        row_indices[k] = i;
        values[k] = x[i] * y.value(j_);
        ++k;
      }
    }
    col_offsets[y.num_non_zeros()] = k;
    for (Index j = 1; j < y.num_non_zeros(); ++j) {
      if (col_offsets[j] == 0)
        col_offsets[j] = col_offsets[j-1];
    }
    r.reset_nocopy(x.size(), y.size(), col_offsets, row_indices, values);
    return r;
  }

  //============================================================================
  // Matrix-Vector operations: implementations
  //============================================================================

  //! Dense matrix  x  sparse vector --> dense vector
  template <typename T, typename Index>
  vector<T> operator*(const matrix<T>& m, const sparse_vector_i<T,Index>& v) {
    assert(m.size2() == v.size());
    vector<T> y(m.size1(),0);
    const T* m_it = m.begin();
    for (Index i = 0; i < y.size(); ++i) {
      y[i] = dot(dense_vector_view<T,Index>(m.size2(), m_it), v);
      m_it += m.size2();
    }
    assert(m_it == m.end());
    return y;
  }

} // namespace sill

#endif // #ifndef _SILL_VECTOR_MATRIX_OPS_HPP_
