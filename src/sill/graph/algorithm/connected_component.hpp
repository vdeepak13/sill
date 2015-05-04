#ifndef SILL_CONNECTED_COMPONENT_HPP
#define SILL_CONNECTED_COMPONENT_HPP

#include <sill/global.hpp>

#include <queue>
#include <tuple>
#include <utility>

namespace sill {

  /**
   * Computes all vertices within a certain distance from a start vertex.
   *
   * \tparam Graph a graph type (directed or undirected).
   * \tparam Set a set type with elements of type Graph::vertex_type
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename Set>
  void connected_component(const Graph& graph,
                           typename Graph::vertex_type root,
                           size_t nhops,
                           Set& result) {
    typedef typename Graph::vertex_type vertex_type;
    typedef typename Graph::edge_type edge_type;

    result.insert(root);
    std::queue<std::pair<vertex_type, size_t> > q; // vertex-distance pairs
    q.push(std::make_pair(root, 0));

    while (!q.empty()) {
      vertex_type u;
      size_t dist;
      std::tie(u, dist) = q.front();
      if (dist >= nhops) break;
      q.pop();
      for (edge_type e : graph.out_edges(u)) {
        vertex_type v = e.target();
        if (!result.count(v)) {
          result.insert(v);
          q.push(std::make_pair(v, dist + 1));
        }
      }
    }
  }

  /**
   * Computes all vertices in the connected component with the given root.
   *
   * \tparam Graph a graph type (directed or undirected).
   * \tparam Set a set type with elements of type Graph::vertex_type
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename Set>
  void connected_component(const Graph& graph,
                           typename Graph::vertex_type root,
                           Set& result) {
    connected_component(graph, root, std::numeric_limits<size_t>::max(), result);
  }

} // namespace sill

#endif
