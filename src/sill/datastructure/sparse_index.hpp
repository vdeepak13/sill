#ifndef SILL_SPARSE_INDEX_HPP
#define SILL_SPARSE_INDEX_HPP

#include <utility>
#include <vector>

namespace sill {

  template <typename T>
  using sparse_index = std::vector<std::pair<std::size_t, T>>;
  
}

#endif
