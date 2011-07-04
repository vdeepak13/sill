
#include <sill/math/sparse_linear_algebra/blas.hpp>
#include <sill/math/sparse_linear_algebra/vector_matrix_ops.hpp>

namespace sill {

  //============================================================================
  // Matrix-Vector operations: implementations
  //============================================================================
  /*
  namespace impl {

    template <>
    vector<double>
    mult_densemat_sparsevec_<sparse_vector<double,size_t>,double,size_t>
    (const matrix<double>& A, const sparse_vector<double,size_t>& x) {
      assert(A.n_cols == x.size());
      vector<double> y(A.n_rows,0);
      int n = A.n_rows;
      int inc = 1;
      for (size_t k = 0; k < x.num_non_zeros(); ++k) {
        double alpha = x.value(k);
        blas::daxpy_(&n, &alpha, A.begin() + A.n_rows * x.index(k), &inc,
                     y.begin(), &inc);
      }
      return y;
    }

  } // namespace impl
  */

  //============================================================================
  // Matrix-Matrix operations: implementations
  //============================================================================

  template <>
  matrix<double>&
  operator+=<double,size_t>
  (matrix<double>& A,
   const rank_one_matrix<vector<double>,sparse_vector<double,size_t> >& B) {
    assert(A.n_rows == B.n_rows && A.n_cols == B.n_cols);
    int n = A.n_rows;
    int inc = 1;
    for (size_t k = 0; k < B.y().num_non_zeros(); ++k) {
      double a = B.y().value(k);
      blas::daxpy_(&n,
                   &a,
                   B.x().begin(),
                   &inc,
                   A.begin() + B.y().index(k) * A.n_rows,
                   &inc);
    }
    return A;
  }

} // namespace sill
