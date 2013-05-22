
#ifndef SILL_ELIMINATION_HPP
#define SILL_ELIMINATION_HPP

#include <vector>

#include <boost/concept_archetype.hpp>

#include <sill/global.hpp>
#include <sill/datastructure/mutable_queue.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Places a clique of edges over all neighbors of a vertex and then
   * removes all edges incident to the vertex.
   *
   * @tparam Graph  E.g., undirected_graph.
   * @param v       a vertex
   * @param g       the graph which is modified by eliminating v
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  void eliminate(typename Graph::vertex v, Graph& g) {
    //concept_assert((EdgeMutableGraph<Graph>));
    //concept_assert((AdjacencyGraph<Graph>));
    // Connect all the neighbors of v.
    g.make_clique(g.neighbors(v));
    // Remove all edges incident to this vertex.
    g.clear_edges(v);
  }

  /**
   * Runs the vertex elimination algorithm on a graph.
   * Eliminating a node from a graph involves connecting the node's neighbors
   * into a new clique and then removing the node from the graph.
   * The node elimination algorithm eliminates all nodes from a graph
   * in some order.
   * The order is chosen greedily using an elimination strategy.
   *
   * Note: This function should be called with a sill:: prefix to avoid
   *       accidental mistakes in name resolution.
   *
   * @tparam Graph
   *         The graph type.  E.g., undirected_graph.
   * @tparam VertexVisitor
   *         A functor type which implements operator()(v, g),
   *         where v is a vertex and g is the graph supplied to this function.
   * @tparam Strategy
   *         The type of elimination strategy; it must model the
   *         EliminationStrategy concept.
   * @param graph
   *        the graph whose nodes are eliminated; it must have no
   *        self-loops or parallel edges, and its edges must be
   *        undirected or bidirected.
   * @param visitor
   *        The visitor which is applied to each vertex before the vertex is
   *        eliminated.
   * @param elim_strategy
   *        the elimination strategy used to determine the elimination order
   *
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename VertexVisitor, typename Strategy>
  void eliminate(Graph& graph, VertexVisitor visitor, Strategy elim_strategy) {
    // concept_assert((EliminationStrategy<Strategy, Graph>));
    // concept_assert((VertexListGraph<Graph>));
    // concept_assert((MutableGraph<Graph>));

    typedef typename Graph::vertex vertex;
    typedef typename Strategy::priority_type priority_type;

    // Make a priority queue of vertex indices, ordered by priority.
    typename sill::mutable_queue<vertex, priority_type> pq;
    foreach(vertex v, graph.vertices()) {
      priority_type priority = elim_strategy.priority(v, graph);
      pq.push(v, priority);
    }

    // Start the vertex elimination loop.
    std::vector<vertex> recompute_priority;
    while (!pq.empty()) {
      // Unqueue the next vertex to be eliminated.
      vertex u = pq.pop().first;
      // Query the elim. strategy for the vertices whose priority may change.
      recompute_priority.clear();
      elim_strategy.updated(u, graph, std::back_inserter(recompute_priority));
      // Visit the vertex.
      visitor(u, graph);
      // Eliminate the vertex.
      eliminate(u, graph);
      // Update the priorities of those vertices whose priority may have changed
      foreach(vertex v, recompute_priority)
        pq.update(v, elim_strategy.priority(v, graph));
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif 
