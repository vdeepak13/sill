
#ifndef _SILL_NORMS_HPP_
#define _SILL_NORMS_HPP_

#warning "norms.hpp is deprecated"

//#include <itpp/stat/misc_stat.h>
//#include <itpp/base/matfunc.h>

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/sparse_linear_algebra/sparse_vector.hpp>

namespace sill {

  template <typename T>
  double norm_inf(const arma::Col<T>& v) {
    return max(abs(v));
  }
  
  template <typename T>
  double norm_inf(const arma::Mat<T>& a) {
    return max(max(abs(a)));
  }

  template <typename T>
  double norm_1(const arma::Col<T>& v) {
    return sum(abs(v));
  }

  template <typename T>
  double norm_1(const arma::Mat<T>& a) {
    return sumsum(abs(a));
  }

  template <typename T>
  double norm_2(const arma::Col<T>& v) {
    return sqrt(dot(v,v));
  }

} // namespace sill

#endif // #ifndef _SILL_NORMS_HPP_