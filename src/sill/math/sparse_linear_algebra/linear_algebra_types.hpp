
#ifndef _SILL_LINEAR_ALGEBRA_TYPES_HPP_
#define _SILL_LINEAR_ALGEBRA_TYPES_HPP_

#include <sill/math/sparse_linear_algebra/sparse_linear_algebra.hpp>

namespace sill {

  /**
   * Specifies dense/sparse linear algebra.
   *
   * This should be used as a template parameter to classes which can be
   * instantiated using either dense or sparse linear algebra.
   */
  enum linear_algebra_enum { DENSE_LINEAR_ALGEBRA, SPARSE_LINEAR_ALGEBRA };

  /**
   * Class which defines vector/matrix types,
   * given a dense/sparse linear algebra specifier.
   *
   * @tparam T      Type of value (e.g., float).
   * @tparam Index  Type of value (e.g., size_t).
   * @tparam LA     Dense/sparse linear algebra specifier.
   */
  template <typename T, typename Index, linear_algebra_enum LA>
  struct linear_algebra_types { };

  //! Specialization for dense linear algebra.
  template <typename T, typename Index>
  struct linear_algebra_types<T, Index, DENSE_LINEAR_ALGEBRA> {

    typedef T          value_type;
    typedef Index      index_type;
    typedef vector<T>  vector_type;
    typedef matrix<T>  matrix_type;

  };

  //! Specialization for sparse linear algebra.
  template <typename T, typename Index>
  struct linear_algebra_types<T, Index, SPARSE_LINEAR_ALGEBRA> {

    typedef T                       value_type;
    typedef Index                   index_type;
    typedef sparse_vector<T,Index>  vector_type;
    typedef sparse_matrix<T,Index>  matrix_type;

  };

} // namespace sill

#endif // #ifndef _SILL_LINEAR_ALGEBRA_TYPES_HPP_
