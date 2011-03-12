
#ifndef _SILL_LINEAR_ALGEBRA_TYPES_HPP_
#define _SILL_LINEAR_ALGEBRA_TYPES_HPP_

#include <sill/math/sparse_linear_algebra/sparse_linear_algebra.hpp>

/**
 * \file linear_algebra_types.hpp  Dense/sparse linear algebra specifications.
 *
 * This file contains structs which are passed to methods as template
 * parameters in order to specify what vector/matrix classes should be used.
 * SILL provides dense and sparse versions, but the user may implement and
 * pass in others.
 *
 * STANDARD: Classes which take a linear algebra specifier as a template
 *           parameter (or have one hard-coded) should typedef the specifier
 *           as "la_type" as a standard name for other classes to use.
 */

namespace sill {

  //! Dense linear algebra specification.
  template <typename T = double, typename Index = size_t>
  struct dense_linear_algebra {

    typedef vector<T>  vector_type;
    typedef matrix<T>  matrix_type;
    typedef typename vector_type::value_type value_type;
    typedef typename vector_type::index_type index_type;

  };

  //! Dense linear algebra specification.
  template <typename T = double, typename Index = size_t>
  struct sparse_linear_algebra {

    typedef sparse_vector<T,Index>  vector_type;
    typedef csc_matrix<T,Index>     matrix_type;
    typedef typename vector_type::value_type value_type;
    typedef typename vector_type::index_type index_type;

  };

  /**
   * Specifies dense/sparse linear algebra.
   *
   * This should be used as a template parameter to classes which can be
   * instantiated using either dense or sparse linear algebra.
   */
//  enum linear_algebra_enum { DENSE_LINEAR_ALGEBRA, SPARSE_LINEAR_ALGEBRA };

  /**
   * Class which defines vector/matrix types,
   * given a dense/sparse linear algebra specifier.
   *
   * @tparam T      Type of value (e.g., float).
   * @tparam Index  Type of value (e.g., size_t).
   * @tparam LA     Dense/sparse linear algebra specifier.
   */
//  template <typename T, typename Index, linear_algebra_enum LA>
//  struct linear_algebra_types { };

/*
  //! Specialization for dense linear algebra.
  template <typename T, typename Index>
  struct linear_algebra_types<T, Index, DENSE_LINEAR_ALGEBRA> {

    typedef T          value_type;
    typedef Index      index_type;
    typedef vector<T>  vector_type;
    typedef matrix<T>  matrix_type;

  };

  //! Specialization for sparse linear algebra.
  //! Note that csc_matrix is the default matrix representation;
  //! classes should be specialized to use other representations as needed.
  template <typename T, typename Index>
  struct linear_algebra_types<T, Index, SPARSE_LINEAR_ALGEBRA> {

    typedef T                       value_type;
    typedef Index                   index_type;
    typedef sparse_vector<T,Index>  vector_type;
    typedef csc_matrix<T,Index>  matrix_type;

  };
*/

} // namespace sill

#endif // #ifndef _SILL_LINEAR_ALGEBRA_TYPES_HPP_
