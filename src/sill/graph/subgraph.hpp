#ifndef SILL_SUBGRAPH_HPP
#define SILL_SUBGRAPH_HPP

#include <queue>
#include <set>

#include <sill/global.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  //! Collect all vertices within a certain count from a start vertex
  //! \ingroup graph_algorithms
  template <typename Graph>
  std::set<typename Graph::vertex>
  connected_component(const Graph& graph,
                      typename Graph::vertex root,
                      size_t nhops) {
    typedef typename Graph::vertex vertex;
    typedef typename Graph::edge edge;

    std::set<vertex> visited;
    visited.insert(root);
    std::queue< std::pair<vertex, size_t> > q; // vertex-distance pairs

    q.push(std::make_pair(root, 0));
    while(!q.empty()) {
      vertex u;
      size_t dist;
      boost::tie(u, dist) = q.front();
      if (dist >= nhops) break;
      q.pop();
      foreach(edge e, graph.out_edges(u)) {
        vertex v = e.target();
        if (!visited.count(v)) {
          visited.insert(v);
          q.push(std::make_pair(v, dist + 1));
        }
      }
    }
    
    return visited;
  }

  //! Computes a subgraph of a graph
  //! \ingroup graph_algorithms
  template <typename Graph, typename VertexRange>
  void subgraph(const Graph& graph,
                const VertexRange& new_vertices,
                Graph& new_graph) {
    typedef typename Graph::vertex vertex;
    typedef typename Graph::edge edge;
    concept_assert((InputRangeConvertible<VertexRange, vertex>));
    
    new_graph.clear();
    foreach(vertex v, new_vertices)
      new_graph.add_vertex(v, graph[v]);

    foreach(vertex v, new_vertices) 
      foreach(edge e, graph.out_edges(v)) 
        if (new_graph.contains(e.target())) 
          new_graph.add_edge(e.source(), e.target(), graph[e]);
  }
                               
  //! Computes a subgraph of a graph
  //! \ingroup graph_algorithms
  template <typename Graph>
  void subgraph(const Graph& graph,
                typename Graph::vertex root,
                size_t nhops,
                Graph& new_graph) {
    subgraph(graph, connected_component(graph, root, nhops), new_graph);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
