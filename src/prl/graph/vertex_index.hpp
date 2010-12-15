#ifndef PRL_VERTEX_INDEX_HPP
#define PRL_VERTEX_INDEX_HPP

#include <boost/unordered_map.hpp>

#include <prl/global.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  template <typename Graph>
  void vertex_index(const Graph& g, 
                    boost::unordered_map<typename Graph::vertex, size_t>& map) {
    size_t i = 0;
    foreach(typename Graph::vertex v, g.vertices()) 
      map[v] = i++;
  }

}

#include <prl/macros_undef.hpp>

#endif
