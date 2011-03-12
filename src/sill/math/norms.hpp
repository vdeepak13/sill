
#ifndef _SILL_NORMS_HPP_
#define _SILL_NORMS_HPP_

#include <itpp/stat/misc_stat.h>
#include <itpp/base/matfunc.h>

namespace sill {

  template <typename T>
  double norm_inf(const itpp::Vec<T>& v) {
    return max(abs(v));
  }
  
  template <typename T>
  double norm_inf(const itpp::Mat<T>& a) {
    return max(max(abs(a)));
  }

  template <typename T>
  double norm_1(const itpp::Vec<T>& v) {
    return sum(abs(v));
  }

  template <typename T>
  double norm_1(const itpp::Mat<T>& a) {
    return sumsum(abs(a));
  }

  template <typename T>
  double norm_2(const itpp::Vec<T>& v) {
    return sqrt(dot(v,v));
  }
  
} // namespace sill

#endif // #ifndef _SILL_NORMS_HPP_
