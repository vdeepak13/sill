#ifndef PRL_MATH_UBLAS_SPECIAL_MATRICES_HPP
#define PRL_MATH_UBLAS_SPECIAL_MATRICES_HPP

#include <prl/math/ublas.hpp>

namespace prl {

  //!\addtogroup math_ublas
  //!@{

  //! Default vector type
  typedef ublas_double::vector_type vec;

  //! Default matrix type
  typedef ublas_double::matrix_type mat;

  //! Returns the identity matrix of the given dimension
  inline ublas::identity_matrix<double> identity(size_t size) {
    return ublas::identity_matrix<double>(size);
  }

  //! Returns a unit vector with the specified index set to 1
  inline ublas::unit_vector<double> unit_vector(size_t size, size_t index) {
    return ublas::unit_vector<double>(size, index);
  }

  //! Returns the zero vector of the specified length
  inline ublas::zero_vector<double> zeros(size_t size) {
    return ublas::zero_vector<double>(size);
  }

  //! Returns the all-zero matrix of the specified dimensions
  inline ublas::zero_matrix<double> zeros(size_t size1, size_t size2) {
    return ublas::zero_matrix<double>(size1, size2);
  }

  //! Returns the all-ones vector of the specified length
  inline ublas::scalar_vector<double> ones(size_t size) {
    return ublas::scalar_vector<double>(size, 1);
  }

  //! Returns the all-one matrix of the specified dimensions
  inline ublas::scalar_matrix<double> ones(size_t size1, size_t size2) {
    return ublas::scalar_matrix<double>(size1, size2, 1);
  }

  //! Returns the all-scalar vector of the specified length
  inline ublas::scalar_vector<double> scalars(size_t size, double value) {
    return ublas::scalar_vector<double>(size, value);
  }

  //! Returns the all-scalar matrix of the specified dimensions
  inline ublas::scalar_matrix<double> scalars(size_t size1, size_t size2,
                                              double value) {
    return ublas::scalar_matrix<double>(size1, size2, value);
  }

  //! Generates n equally spaced points between start and stop.
  inline vec linspace(double start, double stop, size_t n) {
    return ublas_double::linspace(start, stop, n);
  }

  //! An open range of indices
  //! Construct the range with index_range(start, stop), and index_range::all()
  typedef ublas::range index_range;

  //! Returns a vector of length 1
  template <typename T>
  ublas::vector<T> vec_1(T v0) {
    ublas::vector<T> v(1);
    v(0) = v0;
    return v;
  }

  //! Returns a vector of length 2
  template <typename T>
  ublas::vector<T> vec_2(T v0, T v1) {
    ublas::vector<T> v(2);
    v(0) = v0;
    v(1) = v1;
    return v;
  }
  
  //! Returns a vector of length 3
  template <typename T>
  ublas::vector<T> vec_2(T v0, T v1, T v2) {
    ublas::vector<T> v(3);
    v(0) = v0;
    v(1) = v1;
    v(2) = v2;
    return v;
  }


  //! Returns a matrix of size 1x1
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_1(T m00) {
    ublas::matrix<T, ublas::column_major> m(1,1);
    m(0,0) = m00;
    return m;
  }

  //! Returns a matrix of size 1x2
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_1x2(T m00, T m01) {
    ublas::matrix<T, ublas::column_major> m(1,2);
    m(0,0) = m00; m(0,1) = m01;
    return m;
  }

  //! Returns a matrix of size 2x1
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_2x1(T m00,
          T m10) {
    ublas::matrix<T, ublas::column_major> m(2,1);
    m(0,0) = m00;
    m(1,0) = m10;
    return m;
  }

  //! Returns a matrix of size 2x2
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_2x2(T m00, T m01,
          T m10, T m11) {
    ublas::matrix<T, ublas::column_major> m(2,2);
    m(0,0) = m00; m(0,1) = m01;
    m(1,0) = m10; m(1,1) = m11;
    return m;
  }

  //! Returns a matrix of size 1x3
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_1x3(T m00, T m01, T m02) {
    ublas::matrix<T, ublas::column_major> m(1,3);
    m(0,0) = m00; m(0,1) = m01; m(0,2) = m02;
    return m;
  }

  //! Returns a matrix of size 3x1
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_3x1(T m00,
          T m10,
          T m20) {
    ublas::matrix<T, ublas::column_major> m(3,1);
    m(0,0) = m00;
    m(1,0) = m10;
    m(2,0) = m20;
    return m;
  }

  //! Returns a matrix of size 2x3
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_2x3(T m00, T m01, T m02,
          T m10, T m11, T m12) {
    ublas::matrix<T, ublas::column_major> m(2,3);
    m(0,0) = m00; m(0,1) = m01; m(0,2) = m02;
    m(1,0) = m10; m(1,1) = m11; m(1,2) = m12;
    return m;
  }

  //! Returns a matrix of size 3x2
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_3x2(T m00, T m01,
          T m10, T m11,
          T m20, T m21) {
    ublas::matrix<T, ublas::column_major> m(3,2);
    m(0,0) = m00; m(0,1) = m01;
    m(1,0) = m10; m(1,1) = m11;
    m(2,0) = m20; m(2,1) = m21;
    return m;
  }

  //! Returns a matrix of size 2x3
  template <typename T>
  ublas::matrix<T, ublas::column_major> 
  mat_2x3(T m00, T m01, T m02,
          T m10, T m11, T m12,
          T m20, T m21, T m22) {
    ublas::matrix<T, ublas::column_major> m(3,3);
    m(0,0) = m00; m(0,1) = m01; m(0,2) = m02;
    m(1,0) = m10; m(1,1) = m11; m(1,2) = m12;
    m(2,0) = m20; m(2,1) = m21; m(2,2) = m22;
    return m;
  }

  //!@} group math_ublas
}

#endif
