#include <sill/model/decomposable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <>
  void decomposable<table_factor>::possibly_prenormalize() {
    foreach(const vertex& v, vertices()) {
      if (jt[v].is_normalizable())
        jt[v].normalize();
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>
