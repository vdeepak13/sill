
#include <sill/math/sparse_linear_algebra/blas.hpp>
#include <sill/math/sparse_linear_algebra/vector_matrix_ops.hpp>

namespace sill {

  template <>
  matrix<double>&
  operator+=<double,size_t>
  (matrix<double>& A,
   const rank_one_matrix<vector<double>,sparse_vector<double,size_t> >& B) {
    assert(A.size1() == B.size1() && A.size2() == B.size2());
    int n = A.size1();
    int inc = 1;
    for (size_t k = 0; k < B.y().num_non_zeros(); ++k) {
      double a = B.y().value(k);
      blas::daxpy_(&n,
                   &a,
                   B.x().begin(),
                   &inc,
                   A.begin() + B.y().index(k) * A.size1(),
                   &inc);
    }
    return A;
  }

} // namespace sill
