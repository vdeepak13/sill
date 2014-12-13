#ifndef SILL_TEST_TREE_HPP
#define SILL_TEST_TREE_HPP

#include <queue>

#include <sill/functional.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Tests if the connected component containing the root forms
   * a tree.
   * \param filter
   *        a functor taking (e,g) that returns false if an edge e
   *        in graph g is to be ignored
   * \return the number of reachable nodes or 0 if the connected
   *         component is not a tree.
   * \ingroup graph_algorithms
   */
  template <typename Graph, typename EdgeFilter>
  size_t test_tree(const Graph& g, typename Graph::vertex root,
                   EdgeFilter filter){
    typedef typename Graph::vertex vertex;
    typedef typename Graph::edge edge;
    typedef std::pair<vertex, vertex> vertex_parent;
    std::set<vertex> visited;
    visited.insert(root);
    std::queue<vertex_parent> q;
    q.push(vertex_parent(root, vertex()));
    while(!q.empty()) {
      vertex u, parent;
      boost::tie(u, parent) = q.front();
      q.pop();
      foreach(edge e, g.out_edges(u)) {
        vertex v = e.target();
        if (v != parent && filter(e, g)) {
          if (visited.count(v)) return 0;
          visited.insert(v);
          q.push(vertex_parent(v, u));
        }
      }
    }
    return visited.size();
  }
          
  /**
   * Tests if the connected component containing the root forms
   * a tree.
   * \return the number of reachable nodes or 0 if the connected
   *         component is not a tree.
   * \ingroup graph_algorithms
   */
  template <typename Graph>
  size_t test_tree(const Graph& g, typename Graph::vertex v) {
    return test_tree(g, v, make_constant(true));
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
