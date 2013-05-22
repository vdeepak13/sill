#include <iostream>
#include <sstream>
#include <cmath>

#include <sill/math/function/logistic_discrete.hpp>

namespace sill {

  //! Evaluates the function on a discrete input
  double logistic_discrete::operator()(const uvec& x) const {
    assert(x.size() == w.n_rows);
    double arg = b;
    for(size_t i = 0; i < x.size(); i++) {
      assert(x[i] < w.n_cols);
      arg += w(i, x[i]);
    }
    return 1.0 / (1 + std::exp(-arg));
  }

  double logistic_discrete::operator()(const uvec& x, const vec& u) const {
    assert(x.size() == w.n_rows);
    double arg = b;
    for(size_t i = 0; i < x.size(); i++) {
      assert(x[i] < w.n_cols);
      arg += w(i, x[i]) * u[i];
    }
    return 1.0 / (1 + std::exp(-arg));
  }
  
  double logistic_discrete::operator()(const mat& x) const {
    assert(x.n_rows == w.n_rows);
    assert(x.n_cols == w.n_cols);
    double arg = b + dot(w,x);
    return 1.0 / (1 + std::exp(-arg));
  }  
  
  std::ostream& operator<<(std::ostream& out, const logistic_discrete& f) {
    out << f.w << ' ' << f.b;
    return out;
  }

  std::istream& operator>>(std::istream& in, logistic_discrete& f) {
    //    in >> std::ws >> f.w >> std::ws >> f.b;
    in >> std::ws;
    in >> f.w >> std::ws >> f.b;
    return in;
  }

}
