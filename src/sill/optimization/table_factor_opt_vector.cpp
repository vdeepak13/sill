#include <iostream>

#include <sill/optimization/table_factor_opt_vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  std::ostream& operator<<(std::ostream& out,
                           const table_factor_opt_vector& f) {
    if (f.f.size() < 100)
      out << f.f;
    else
      out << "table_factor_opt_vector[size=" << f.f.size() << "]";
    return out;
  }

}  // namespace sill

#include <sill/macros_undef.hpp>
