
#include <sill/optimization/logreg_opt_vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <>
  void ov_axpy<logreg_opt_vector<double> >(double a,
                                           const logreg_opt_vector<double>& x,
                                           logreg_opt_vector<double>& y) {
    y.f += a * x.f;
    y.v += a * x.v;
    y.b += a * x.b;
  }

  template <>
  void ov_axpy<logreg_opt_vector<float> >(double a,
                                          const logreg_opt_vector<float>& x,
                                          logreg_opt_vector<float>& y) {
    y.f += a * x.f;
    y.v += a * x.v;
    y.b += a * x.b;
  }

}  // namespace sill
