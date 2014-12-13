#ifndef SILL_ANCESTORS_HPP
#define SILL_ANCESTORS_HPP

#include <set>
#include <queue>

#include <sill/graph/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Returns the ancestors for a set of vertices. 
   * \ingroup graph_algorithms.
   */
  template <typename Graph>
  std::set<typename Graph::vertex> 
  ancestors(const std::set<typename Graph::vertex>& vertices, const Graph& graph) {
    typedef typename Graph::vertex vertex;
    std::set<vertex> result;
    std::queue<vertex> q;
    foreach(vertex v, vertices) q.push(v);
    while(!q.empty()) {
      vertex u = q.front(); q.pop();
      foreach(vertex v, graph.parents(u))
        if(!result.count(v)) {
          result.insert(v);
          q.push(v);
        }
    }
    return result;
  }

  /**
   * Returns the ancestors for a vertex.
   * \ingroup graph_algorithms.
   */
  template <typename Graph>
  std::set<typename Graph::vertex> 
  ancestors(const typename Graph::vertex& v, const Graph& graph) {
    std::set<typename Graph::vertex> vset;
    vset.insert(v);
    return ancestors(vset, graph);
  }

}

#include <sill/macros_undef.hpp>

#endif
