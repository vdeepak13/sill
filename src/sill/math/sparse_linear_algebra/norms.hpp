
#ifndef _SILL_SPARSE_LINEAR_ALGEBRA_NORMS_HPP_
#define _SILL_SPARSE_LINEAR_ALGEBRA_NORMS_HPP_

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/sparse_linear_algebra/sparse_vector.hpp>

namespace sill {

  template <typename T, typename Index>
  double norm_inf(const sparse_vector<T,Index>& v) {
    return max(abs(v));
  }

  template <typename T, typename Index>
  double norm_1(const sparse_vector<T,Index>& v) {
    return v.L1norm();
  }

  template <typename T, typename Index>
  double norm_2(const sparse_vector<T,Index>& v) {
    return v.L2norm();
  }
  
} // namespace sill

#endif // #ifndef _SILL_SPARSE_LINEAR_ALGEBRA_NORMS_HPP_
