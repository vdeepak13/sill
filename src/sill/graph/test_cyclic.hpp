#ifndef SILL_TEST_TREE_HPP
#define SILL_TEST_TREE_HPP

#include <sill/datastructure/mutable_queue.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Test if a directed graph is cyclic.
   *
   * @return <bool, vertex>, where the bool is true iff the graph is cyclic.
   *         (The bool is false for empty graphs.)  If a cycle is found, then
   *         the vertex is set to a vertex within the cycle.
   *
   * \ingroup graph_algorithms
   */
  template <typename DirectedGraph>
  std::pair<bool, typename DirectedGraph::vertex>
  test_cyclic(const DirectedGraph& g) {

    typedef typename DirectedGraph::vertex vertex;
    typedef typename DirectedGraph::edge edge;

    mutable_queue<vertex, double> q;
    foreach(const vertex& v, g.vertices()) {
      q.push(v, - (double)(g.in_degree(v)));
    }

    while (!q.empty()) {
      std::pair<vertex, double> v_indeg(q.pop());
      // If all remaining vertices have parents, then there is a cycle.
      if (v_indeg.second != 0) {
        return std::make_pair(true, v_indeg.first);
      }
      // Remove edges from v to children.
      foreach(const vertex& child, g.children(v_indeg.first)) {
        q.increment_if_present(child, 1);
      }
    }
    return std::make_pair(false, g.null_vertex());
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
