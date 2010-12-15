#include <iostream>
#include <cmath>

#include <prl/math/function/logistic_discrete.hpp>
#include <prl/math/linear_algebra.hpp>

namespace prl {

  //! Evaluates the function on a discrete input
  double logistic_discrete::operator()(const ivec& x) const {
    assert(x.size() == w.size1());
    double arg = b;
    for(size_t i = 0; i < x.size(); i++) {
      assert(x[i] >= 0 && size_t(x[i]) < w.size2());
      arg += w(i, x[i]);
    }
    return 1.0 / (1 + std::exp(-arg));
  }

  double logistic_discrete::operator()(const ivec& x, const vec& u) const {
    assert(x.size() == w.size1());
    double arg = b;
    for(size_t i = 0; i < x.size(); i++) {
      assert(x[i] >=0 && size_t(x[i]) < w.size2());
      arg += w(i, x[i]) * u[i];
    }
    return 1.0 / (1 + std::exp(-arg));
  }
  
  double logistic_discrete::operator()(const mat& x) const {
    assert(x.size1() == w.size1());
    assert(x.size2() == w.size2());
    double arg = b + sumsum(elem_mult(w, x));
    return 1.0 / (1 + std::exp(-arg));
  }  
  
  std::ostream& operator<<(std::ostream& out, const logistic_discrete& f) {
    out << f.w << ' ' << f.b;
    return out;
  }

  std::istream& operator>>(std::istream& in, logistic_discrete& f) {
    in >> std::ws >> f.w >> std::ws >> f.b;
    return in;
  }

}
