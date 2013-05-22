
#include <sill/base/vector_assignment.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Returns the size of the vector variables in this map.
  //! \relates vector_variable
  size_t vector_size(const vector_assignment& va) {
    size_t size(0);
    typedef std::pair<vector_variable*, vec> map_key_type;
    foreach(const map_key_type& val, va)
      size += val.first->size();
    return size;
  }

  //! @}
}

#include <sill/macros_undef.hpp>
