
#ifndef SILL_GRAPH_TRAVERSAL_HPP
#define SILL_GRAPH_TRAVERSAL_HPP

#include <deque>
#include <vector>
#include <iterator>
#include <map>

#include <sill/global.hpp>
#include <sill/graph/algorithm/output_edge_visitor.hpp>
#include <sill/range/reversed.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Visits each vertex of a directed acyclic graph once in a traversal
   * such that each \f$v\f$ is visited after all nodes \f$u\f$ with
   * \f$u \rightarrow v\f$ are visited.
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  std::vector<typename Graph::vertex>
  directed_partial_vertex_order(const Graph& g) {
    typedef typename Graph::edge edge;
    typedef typename Graph::vertex vertex;

    if (g.num_vertices() == 0)
      return std::vector<vertex>();

    // Find all vertices without parents
    //   Maintain a set of 'remaining_edges' which still need to be traversed,
    //   the sources of which are in 'vertices' and the targets of which
    //   may be in 'vertices' or 'remaining_vertices.'
    std::list<edge> remaining_edges;
    std::vector<vertex> vertices;
    // Contains remaining vertices, plus a count of the number of in-edges
    // in remaining_edges.
    std::map<vertex, size_t> remaining_vertices;
    foreach(vertex v, g.vertices()) {
      typename Graph::neighbor_iterator parents_it, parents_end;
      boost::tie(parents_it, parents_end) = g.parents(v);
      if (parents_it == parents_end) {
        vertices.push_back(v);
        foreach(edge e, g.out_edges(v))
          remaining_edges.push_back(e);
      } else
        remaining_vertices[v] = g.in_degree(v);
    }
    assert(vertices.size() > 0);

    // Add in the remaining vertices in partial order
    while(remaining_edges.size() > 0) {
      edge e(remaining_edges.front());
      remaining_edges.pop_front();
      vertex v(e.target());
      if (remaining_vertices.count(v)) {
        size_t in_d(remaining_vertices[v]);
        if (in_d == 1) {
          remaining_vertices.erase(v);
          vertices.push_back(v);
          foreach(edge e2, g.out_edges(v))
            remaining_edges.push_back(e2);
        } else
          remaining_vertices[v] = in_d - 1;
      }
    }

    return vertices;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_GRAPH_TRAVERSAL_HPP
