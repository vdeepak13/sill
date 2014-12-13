#ifndef SILL_VERTEX_INDEX_HPP
#define SILL_VERTEX_INDEX_HPP

#include <boost/unordered_map.hpp>

#include <sill/global.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename Graph>
  void vertex_index(const Graph& g, 
                    boost::unordered_map<typename Graph::vertex, size_t>& map) {
    size_t i = 0;
    foreach(typename Graph::vertex v, g.vertices()) 
      map[v] = i++;
  }

}

#include <sill/macros_undef.hpp>

#endif
