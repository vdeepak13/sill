#ifndef PRL_ANCESTORS_DESCENDANTS_HPP
#define PRL_ANCESTORS_DESCENDANTS_HPP
#include <set>
#include <queue>

#include <prl/graph/concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

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
   * Returns the descendants
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  std::set<typename Graph::vertex>
  descendants(const std::set<typename Graph::vertex>& vertices, const Graph& graph) {
    typedef typename Graph::vertex vertex;
    std::set<vertex> result;
    std::queue<vertex> q;
    foreach(vertex v, vertices) q.push(v);
    while(!q.empty()) {
      vertex u = q.front(); q.pop();
      foreach(vertex v, graph.children(u))
        if(!result.count(v)) {
          result.insert(v);
          q.push(v);
        }
    }
    return result;
  }

}

#include <prl/macros_undef.hpp>

#endif
