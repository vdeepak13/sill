
#include <iostream>

#include <prl/optimization/table_factor_opt_vector.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  std::ostream& operator<<(std::ostream& out,
                           const table_factor_opt_vector& f) {
    if (f.f.size() < 100)
      out << f.f;
    else
      out << "table_factor_opt_vector[size=" << f.f.size() << "]";
    return out;
  }

}  // namespace prl

#include <prl/macros_undef.hpp>
