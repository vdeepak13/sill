
#include <sill/math/linear_algebra/blas.hpp>
#include <sill/math/linear_algebra/vector_matrix_ops.hpp>

namespace sill {

  //============================================================================
  // Matrix-Matrix operations: implementations
  //============================================================================

  template <>
  arma::Mat<double>&
  operator+=<double,arma::u32>
  (arma::Mat<double>& A,
   const rank_one_matrix<arma::Col<double>,sparse_vector<double,arma::u32> >&
   B) {
    assert(A.n_rows == B.n_rows && A.n_cols == B.n_cols);
    int n = A.n_rows;
    int inc = 1;
    for (arma::u32 k = 0; k < B.y().num_non_zeros(); ++k) {
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
