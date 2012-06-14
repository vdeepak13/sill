#ifndef SILL_LINEAR_ALGEBRA_DECOMPOSITIONS_HPP
#define SILL_LINEAR_ALGEBRA_DECOMPOSITIONS_HPP

/**
 * \file decompositions.hpp  Includes dense linear algebra headers.
 *
 * @todo Change this file's name to dense_linear_algebra.hpp?
 */

#include <sill/math/linear_algebra/armadillo.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Estimate the maximum eigenvalue of a matrix A via power iteration.
   *
   * @todo Add convergence criteria, not just an iteration count.
   */
  template <typename MatrixType>
  double power_iteration(const MatrixType& A, size_t iterations) {
    ASSERT_EQ(A.n_rows, A.n_cols);
    vec b(ones(A.n_cols));
    b /= norm(b,2);
    for (size_t i = 0; i < iterations; ++i) {
      b = A * b;
      b /= norm(b,2);
    }
    return dot(b, A * b);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LINEAR_ALGEBRA_DECOMPOSITIONS_HPP
