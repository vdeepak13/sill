#ifndef SILL_SERIALIZE_EIGEN_HPP
#define SILL_SERIALIZE_EIGEN_HPP

#include <sill/math/eigen/dynamic.hpp>
#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

#include <Eigen/Core>

namespace sill {

  //! Serializes a dynamic Eigen vector. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const dynamic_vector<T>& vec) {
    ar << vec.rows();
    for (size_t i = 0; i < vec.rows(); ++i) {
      ar << vec[i];
    }
    return ar;
  }

  //! Serializes a dynamic Eigen matrix. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const dynamic_matrix<T>& mat) {
    ar << mat.rows();
    ar << mat.cols();
    const T* it = mat.data();
    const T* end = it + mat.size();
    for (; it != end; ++it) {
      ar << *it;
    }
    return ar;
  }

  //! Deserializes a dynamic Eigen vector. \relates oarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, dynamic_vector<T>& vec) {
    size_t rows;
    ar >> rows;
    vec.resize(rows);
    for (size_t i = 0; i < vec.rows(); ++i) {
      ar >> vec[i];
    }
    return ar;
  }

  //! Serializes a dynamic Eigen matrix. \relates oarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, dynamic_matrix<T>& mat) {
    size_t rows, cols;
    ar >> rows >> cols;
    mat.resize(rows, cols);
    T* it = mat.data();
    T* end = it + mat.size();
    for (; it != end; ++it) {
      ar >> *it;
    }
    return ar;
  }
  
}

#endif
