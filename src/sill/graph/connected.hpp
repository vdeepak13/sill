#ifndef SILL_GRAPH_CONNECTED_HPP
#define SILL_GRAPH_CONNECTED_HPP

#include <queue>
#include <map>
#include <set>
#include <sill/macros_def.hpp>

namespace sill {

  //! Returns true if an undirected graph is connected
  //! \ingroup graph_algorithms
  template <typename Graph>
  bool is_connected(const Graph& graph) {
    // TODO: concept checking
    typedef typename Graph::vertex vertex;
    if (graph.empty()) return true;

    // Search, starting from some arbitrarily chosen vertex
    vertex root = *graph.vertices().first;
    std::set<vertex> visited;  // unordered_set would be nice here
    visited.insert(root);
    std::queue<vertex> q;
    q.push(root);

    while(!q.empty() && visited.size() < graph.num_vertices()) {
      vertex u = q.front();
      q.pop();
      foreach(vertex v, graph.neighbors(u)) {
        if(!visited.count(v)) {
          visited.insert(v);
          q.push(v);
        }
      }
    }

    return visited.size() == graph.num_vertices();
  }
}

#include <sill/macros_undef.hpp>

#endif
