
#ifndef SILL_TRIANGULATION_HPP
#define SILL_TRIANGULATION_HPP

#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <set>

#include <sill/datastructure/set_index.hpp>
#include <sill/graph/elimination.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A visitor that outputs elimination cliques into an iterator.
   * \ingroup graph_types
   */
  template <typename Graph, typename OutIt>
  struct triangulation_visitor {
    
    typedef typename Graph::vertex vertex;

    triangulation_visitor(OutIt out, bool filter_nonmaximal)
      : out(out), filter_nonmaximal(filter_nonmaximal) { }

    void operator()(vertex v, const Graph& g) {
      // Right now, no attempt is made to convert vertex descriptors to nodes
      std::set<vertex> clique;
      foreach(const vertex &u, g.neighbors(v)) clique.insert(u);
      clique.insert(v);
      // Report the clique (perhaps after a maximality check).
      if (filter_nonmaximal) {
        if (clique_index.is_maximal(clique)) {
          // The clique is maximal; report it and record it.
          *out = clique;
          ++out;
          clique_index.insert(clique);
        }
      } else {
        *out = clique;
        ++out;
      }
    }

  private:
    OutIt out;
    bool filter_nonmaximal;
    set_index< std::set<vertex> > clique_index;
  };

  /**
   * Computes a triangulation of a graph using greedy vertex
   * elimination.
   *
   * @param Graph
   *        The graph type.  It must model the Boost AdjacencyGraph or
   *        AdjacencyMatrix concepts; it must also model the
   *        VertexListGraph concept.  In addition, the graph must have
   *        an interior vertex_index property.
   * @param OutIt
   *        an output iterator type which can be assigned lvalues
   *        of type sill::set<vertex_t>, where vertex_t is
   *        boost::graph_traits<Graph>::vertex_descriptor.
   * @param ElimStrategy
   *        The type of elimination strategy; it must model the
   *        elimination_strategy concept.
   * @param graph
   *        the graph to be triangulated; it must have no self-loops
   *        or parallel edges, and its edges must be undirected or
   *        bidirected.  This procedure removes all edges from the
   *        graph.
   * @param output
   *        the output iterator to which the cliques are assigned
   * @param elim_strategy
   *        the elimination strategy used to determine the elimination
   *        order
   * @param filter_nonmaximal
   *        if this flag is set to true (default), then all reported
   *        cliques are maximal; if it is set to false, then some
   *        cliques may be subsets of other cliques
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph,
            typename OutIt,
            typename ElimStrategy>
  void triangulate(Graph& graph,
                   OutIt output,
                   ElimStrategy elim_strategy = ElimStrategy(),
                   bool filter_nonmaximal = true) {
    triangulation_visitor<Graph, OutIt>
      visitor(output, filter_nonmaximal);
    eliminate(graph, visitor, elim_strategy);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_TRIANGULATION_HPP
