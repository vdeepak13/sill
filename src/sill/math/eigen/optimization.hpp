#ifndef SILL_EIGEN_OPTIMIZATION_HPP
#define SILL_EIGEN_OPTIMIZATION_HPP

#include <sill/traits/vector_value.hpp>

#include <Eigen/Core>

namespace sill {

  //! Specialization of vector_value for Eigen's vector classes.
  template <typename T, int Rows, int Cols>
  struct vector_value<Eigen::Matrix<T, Rows, Cols>> {
    typedef T type;
  };

} // namespace sill

namespace Eigen {

  //! Implements elementwise division.
  template <typename Derived>
  MatrixBase<Derived>&
  operator/=(MatrixBase<Derived>& x, const MatrixBase<Derived>& y) {
    x.array() /= y.array();
    return x;
  }

  //! Implements dot product as a free function.
  template <typename Derived>
  typename Derived::Scalar
  dot(const MatrixBase<Derived>& x, const MatrixBase<Derived>& y) {
    return x.dot(y);
  }

  //! Implements weighted update.
  template <typename Derived>
  void update(MatrixBase<Derived>& x, const MatrixBase<Derived>& y,
              typename Derived::Scalar a) {
    x += a * y;
  }
  
} // namespace Eigen

#endif
