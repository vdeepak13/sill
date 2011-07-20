
/**
 * \file sparse_linear_algebra.hpp  Includes all sparse linear algebra headers.
 */

#ifndef _SILL_SPARSE_LINEAR_ALGEBRA_HPP_
#define _SILL_SPARSE_LINEAR_ALGEBRA_HPP_

#include <sill/math/linear_algebra/coo_matrix.hpp>
#include <sill/math/linear_algebra/csc_matrix.hpp>
#include <sill/math/linear_algebra/sparse_vector.hpp>
#include <sill/math/linear_algebra/vector_matrix_ops.hpp>
#include <sill/math/linear_algebra/norms.hpp>

namespace sill {

  template <typename VecType>
  inline
  VecType zeros(typename VecType::size_type n) {
    VecType v(n);
    v.zeros();
    return v;
  }

} // namespace sill

#endif // #ifndef _SILL_SPARSE_LINEAR_ALGEBRA_HPP_
