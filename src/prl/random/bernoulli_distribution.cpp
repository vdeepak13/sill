#include <iostream>

#include <sill/random/bernoulli_distribution.hpp>

namespace sill {

  std::ostream& operator<<(std::ostream& out, const bernoulli_distribution& d) {
    out << d.p_;
    return out;
  }

  std::istream& operator>>(std::istream& in, bernoulli_distribution& d) {
    in >> std::ws >> d.p_;
    return in;
  }

} // namespace sill
