
/**
 * \file sparse_linear_algebra.hpp  Includes all sparse linear algebra headers.
 */

#ifndef _SILL_SPARSE_LINEAR_ALGEBRA_HPP_
#define _SILL_SPARSE_LINEAR_ALGEBRA_HPP_

#include <sill/math/linear_algebra/coo_matrix.hpp>
#include <sill/math/linear_algebra/csc_matrix.hpp>
#include <sill/math/linear_algebra/norms.hpp>
#include <sill/math/linear_algebra/sparse_vector.hpp>
#include <sill/math/linear_algebra/vector_matrix_ops.hpp>

namespace sill {

  /**
   * Sparse linear algebra specification.
   *
   * This type of struct can be passed to methods as a template parameter
   * to specify what vector/matrix classes should be used.
   *
   * STANDARD: Classes which take a linear algebra type specifier as a template
   *           parameter (or have one hard-coded) should typedef the specifier
   *           as "la_type" as a standard name for other classes to use.
   */
  template <typename T = double, typename SizeType = arma::u32>
  struct sparse_linear_algebra {

    typedef sparse_vector<T,SizeType>  vector_type;
    typedef csc_matrix<T,SizeType>     matrix_type;
    typedef typename vector_type::value_type value_type;
    typedef typename vector_type::size_type  size_type;

    typedef arma::Col<T>  dense_vector_type;
    typedef arma::Mat<T>  dense_matrix_type;

    typedef uvec  index_vector_type;

  };


  template <typename VecType>
  inline
  VecType zeros(typename VecType::size_type n) {
    VecType v(n);
    v.zeros();
    return v;
  }

} // namespace sill

#endif // #ifndef _SILL_SPARSE_LINEAR_ALGEBRA_HPP_
