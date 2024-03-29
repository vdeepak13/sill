#include <iostream>
#include <cmath>
#include <cassert>

#include <sill/math/multinomial_distribution.hpp>

namespace sill {

  multinomial_distribution::multinomial_distribution(const vec& p) : p_(p) {
    using std::abs;
    assert(as_scalar(prod(p >= 0)) == 1);
    assert(fabs(sum(p) - 1) < 1e-8);
  }

  double multinomial_distribution::mean() const {
    double x = 0;
    for(size_t i = 0; i < p_.size(); i++) 
      x += i*p_[i];
    return x;
  }

  std::ostream& operator<<(std::ostream& out, const multinomial_distribution& d)
  {
    out << d.p_;
    return out;
  }

  std::istream& operator>>(std::istream& in, multinomial_distribution& d) {
    in >> std::ws >> d.p_;
    return in;
  }

} // namespace sill
