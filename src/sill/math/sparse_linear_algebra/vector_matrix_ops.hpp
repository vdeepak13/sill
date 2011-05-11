
#ifndef _SILL_VECTOR_MATRIX_OPS_HPP_
#define _SILL_VECTOR_MATRIX_OPS_HPP_

#include <sill/math/matrix.hpp>
#include <sill/math/sparse_linear_algebra/blas.hpp>
#include <sill/math/sparse_linear_algebra/coo_matrix.hpp>
#include <sill/math/sparse_linear_algebra/csc_matrix.hpp>
#include <sill/math/sparse_linear_algebra/rank_one_matrix.hpp>
#include <sill/math/sparse_linear_algebra/sparse_vector.hpp>
#include <sill/math/vector.hpp>

/**
 * \file vector_matrix_ops.hpp  Free functions for vectors and matrices.
 *
 * File contents:
 */

namespace sill {

  /*****************************************************************************
   * Vector-Scalar operations
   *  - operator*
   ****************************************************************************/

  //! const * sparse vector --> sparse vector
  template <typename T, typename SizeType>
  sparse_vector<T,SizeType> operator*(T c, const sparse_vector<T,SizeType>& v);

  /*****************************************************************************
   * Vector-Vector operations
   *  - operator+=
   *  - operator-=
   *  - operator/=
   *  - dot
   *  - outer_product
   *  - elem_mult_out
   *  - elem_square_out
   ****************************************************************************/

  //! Addition
  template <typename T, typename SizeType>
  vector<T>& operator+=(vector<T>& x, const sparse_vector<T,SizeType>& y);

  //! Subtraction
  template <typename T, typename SizeType>
  sparse_vector<T,SizeType>&
  operator-=(sparse_vector<T,SizeType>& x, const vector<T>& y);

  //! Division
  //! WARNING: This ignores zero elements of x. If y has zeros,
  //!          this may ignore values 0 / 0.
  template <typename T, typename SizeType>
  sparse_vector<T,SizeType>&
  operator/=(sparse_vector<T,SizeType>& x, const vector<T>& y);

  //! Dot product.
  template <typename T, typename SizeType>
  T dot(const vector<T>& x, const sparse_vector<T,SizeType>& y);

  //! Dot product.
  template <typename T, typename SizeType>
  T dot(const vector<T>& x, const sparse_vector_view<T,SizeType>& y);

  //! Dot product.
  template <typename T, typename SizeType>
  T dot(const dense_vector_view<T,SizeType>& x, const sparse_vector<T,SizeType>& y);

  //! Dot product.
  template <typename T, typename SizeType>
  T dot(const dense_vector_view<T,SizeType>& x,
        const sparse_vector_view<T,SizeType>& y);

  //! Outer product
  template <typename T, typename SizeType>
  rank_one_matrix<vector<T>, sparse_vector<T,SizeType> >
  outer_product(const vector<T>& x, const sparse_vector<T,SizeType>& y);

  //! Store result of elem_mult(a,b) in c.
  template <typename T, typename SizeType>
  void elem_mult_out(const sparse_vector<T,SizeType>& a,
                     const sparse_vector<T,SizeType>& b,
                     sparse_vector<T,SizeType>& c);

  //! Store result of elem_mult(a,a) in b.
  template <typename T, typename SizeType>
  void elem_square_out(const sparse_vector<T,SizeType>& a,
                       sparse_vector<T,SizeType>& b);

  /*****************************************************************************
   * Matrix-Vector operations
   *  - operator*
   ****************************************************************************/

  //! Dense matrix  x  sparse vector --> dense vector
  template <typename T, typename SizeType>
  vector<T>
  operator*(const matrix<T>& m, const sparse_vector<T,SizeType>& v);

  //! Dense matrix  x  sparse vector --> dense vector
  template <typename T, typename SizeType>
  vector<T>
  operator*(const matrix<T>& m, const sparse_vector_view<T,SizeType>& v);

  /*****************************************************************************
   * Matrix-Matrix operations
   *  - operator+=
   ****************************************************************************/

  //! Dense matrix += rank-one matrix
  template <typename T, typename SizeType>
  matrix<T>&
  operator+=(matrix<T>& A,
             const rank_one_matrix<vector<T>,sparse_vector<T,SizeType> >& B);


  //============================================================================
  // Vector-Scalar operations: implementations
  //============================================================================

  template <typename T, typename SizeType>
  sparse_vector<T,SizeType> operator*(T c, const sparse_vector<T,SizeType>& v) {
    sparse_vector<T,SizeType> r(v);
    r *= c;
    return r;
  }

  //============================================================================
  // Vector-Vector operations: implementations
  //============================================================================

  template <typename T, typename SizeType>
  vector<T>& operator+=(vector<T>& x, const sparse_vector<T,SizeType>& y) {
    assert(x.size() == y.size());
    for (SizeType k = 0; k < y.num_non_zeros(); ++k)
      x[y.index(k)] += y.value(k);
    return x;
  }

  template <typename T, typename SizeType>
  sparse_vector<T,SizeType>&
  operator-=(sparse_vector<T,SizeType>& x, const vector<T>& y) {
    assert(x.size() == y.size());
    // Attempt to keep x sparse.
    std::vector<SizeType> inds;
    std::vector<T> vals;
    for (SizeType i = 0; i < y.size(); ++i) {
      T val = x[i] - y[i];
      if (val != 0) {
        inds.push_back(i);
        vals.push_back(val);
      }
    }
    x = sparse_vector<T,SizeType>(y.size(), inds, vals);
    return x;
  }

  template <typename T, typename SizeType>
  sparse_vector<T,SizeType>&
  operator/=(sparse_vector<T,SizeType>& x, const vector<T>& y) {
    for (SizeType k = 0; k < x.num_non_zeros(); ++k)
      x.value(k) /= y[x.index(k)];
    return x;
  }

  template <typename T, typename SizeType>
  T dot(const vector<T>& x, const sparse_vector<T,SizeType>& y) {
    assert(x.size() == y.size());
    T r = 0;
    for (SizeType i = 0; i < y.num_non_zeros(); ++i)
      r += x[y.index(i)] * y.value(i);
    return r;
  }

  template <typename T, typename SizeType>
  T dot(const vector<T>& x, const sparse_vector_view<T,SizeType>& y) {
    assert(x.size() == y.size());
    T r = 0;
    for (SizeType i = 0; i < y.num_non_zeros(); ++i)
      r += x[y.index(i)] * y.value(i);
    return r;
  }

  template <typename T, typename SizeType>
  T dot(const dense_vector_view<T,SizeType>& x,
        const sparse_vector<T,SizeType>& y) {
    assert(x.size() == y.size());
    T r = 0;
    for (SizeType i = 0; i < y.num_non_zeros(); ++i)
      r += x[y.index(i)] * y.value(i);
    return r;
  }

  template <typename T, typename SizeType>
  T dot(const dense_vector_view<T,SizeType>& x,
        const sparse_vector_view<T,SizeType>& y) {
    assert(x.size() == y.size());
    T r = 0;
    for (SizeType i = 0; i < y.num_non_zeros(); ++i)
      r += x[y.index(i)] * y.value(i);
    return r;
  }

  //! Outer product
  template <typename T, typename SizeType>
  rank_one_matrix<vector<T>, sparse_vector<T,SizeType> >
  outer_product(const vector<T>& x, const sparse_vector<T,SizeType>& y) {
    return make_rank_one_matrix(x,y);
  }

  /* // OLD VERSION
  template <typename T, typename SizeType>
  csc_matrix<T,SizeType> outer_product(const dense_vector_view<T,SizeType>& x,
                                    const sparse_vector_i<T,SizeType>& y) {
    csc_matrix<T,SizeType> r;
    vector<SizeType> col_offsets(y.num_non_zeros() + 1);
    vector<SizeType> row_indices(x.size() * y.num_non_zeros());
    vector<T> values(row_indices.size());
    SizeType k = 0;
    for (SizeType j_ = 0; j_ < y.num_non_zeros(); ++j_) {
      SizeType j = y.index(j_);
      col_offsets[j] = k;
      for (SizeType i = 0; i < x.size(); ++i) {
        row_indices[k] = i;
        values[k] = x[i] * y.value(j_);
        ++k;
      }
    }
    col_offsets[y.num_non_zeros()] = k;
    for (SizeType j = 1; j < y.num_non_zeros(); ++j) {
      if (col_offsets[j] == 0)
        col_offsets[j] = col_offsets[j-1];
    }
    r.reset_nocopy(x.size(), y.size(), col_offsets, row_indices, values);
    return r;
  }
  */

  template <typename T, typename SizeType>
  void elem_mult_out(const sparse_vector<T,SizeType>& a,
                     const sparse_vector<T,SizeType>& b,
                     sparse_vector<T,SizeType>& c) {
    assert(a.size() == b.size());
    if (&a == &b) {
      elem_square_out(a, c);
      return;
    }
    // TO DO: If a,b are sorted, make this more efficient.
    std::vector<SizeType> inds;
    std::vector<T> vals;
    if (a.num_non_zeros() < b.num_non_zeros()) {
      for (SizeType k = 0; k < a.num_non_zeros(); ++k) {
        if (b[a.index(k)] != 0) {
          inds.push_back(a.index(k));
          vals.push_back(a.value(k) * b[a.index(k)]);
        }
      }
    } else {
      for (SizeType k = 0; k < b.num_non_zeros(); ++k) {
        if (a[b.index(k)] != 0) {
          inds.push_back(b.index(k));
          vals.push_back(b.value(k) * a[b.index(k)]);
        }
      }
    }
    c.reset(a.size(), inds, vals);
  }

  template <typename T, typename SizeType>
  void elem_square_out(const sparse_vector<T,SizeType>& a,
                       sparse_vector<T,SizeType>& b) {
    b.resize(a.size(), a.num_non_zeros());
    elem_mult_out(a.values(), a.values(), b.values());
  }

  //============================================================================
  // Matrix-Vector operations: implementations
  //============================================================================

  namespace impl {

    /**
     * Internal dense matrix x sparse vector.
     * If A has size [m, n] and x has size n and k non-zeros,
     * this does m * k multiplications.
     *
     * This version computes one element of y at a time.
     * (It is faster when y is very short.)
     */
    template <typename InVecType, typename T, typename SizeType>
    inline vector<T>
    mult_densemat_sparsevec_(const matrix<T>& A, const InVecType& x) {
      assert(A.size2() == x.size());
      vector<T> y(A.size1(),0);
      const T* A_it = A.begin();
      for (SizeType i = 0; i < y.size(); ++i) {
        y[i] = dot(dense_vector_view<T,SizeType>(A.size2(), A_it, A.size1()),
                   x);
        ++A_it;
      }
      return y;
    }

    /*
    // Specialization
    // TO DO: Use this when y is reasonably long and x is reasonably sparse.
    template <>
    inline vector<double>
    mult_densemat_sparsevec_<sparse_vector<double,size_t>,double,size_t>
    (const matrix<double>& A, const sparse_vector<double,size_t>& x) {
      assert(A.size2() == x.size());
      vector<double> y(A.size1(),0);
      int n = A.size1();
      int inc = 1;
      for (size_t k = 0; k < x.num_non_zeros(); ++k) {
        double alpha = x.value(k);
        blas::daxpy_(&n, &alpha, A.begin() + A.size1() * x.index(k), &inc,
                     y.begin(), &inc);
      }
      return y;
    }
    */

  } // namespace impl

  template <typename T, typename SizeType>
  vector<T> operator*(const matrix<T>& A, const sparse_vector<T,SizeType>& x) {
    return
      impl::mult_densemat_sparsevec_<sparse_vector<T,SizeType>,T,SizeType>
      (A, x);
  }

  //! Dense matrix  x  sparse vector --> dense vector
  template <typename T, typename SizeType>
  vector<T>
  operator*(const matrix<T>& A, const sparse_vector_view<T,SizeType>& x) {
    return
      impl::mult_densemat_sparsevec_<sparse_vector_view<T,SizeType>,T,SizeType>
      (A,x);
  }

  //============================================================================
  // Matrix-Matrix operations: implementations
  //============================================================================

  template <typename T, typename SizeType>
  matrix<T>&
  operator+=(matrix<T>& A,
             const rank_one_matrix<vector<T>,sparse_vector<T,SizeType> >& B) {
    assert(A.size1() == B.size1() && A.size2() == B.size2());
    for (SizeType k = 0; k < B.y().num_non_zeros(); ++k)
      A.add_column(B.y().index(k), B.x() * B.y().value(k));
    return A;
  }

  // Specialization
  template <>
  matrix<double>&
  operator+=<double,size_t>
  (matrix<double>& A,
   const rank_one_matrix<vector<double>,sparse_vector<double,size_t> >& B);

} // namespace sill

#endif // #ifndef _SILL_VECTOR_MATRIX_OPS_HPP_
