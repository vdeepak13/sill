#ifndef PRL_GRAPH_CLIQUE_HPP
#define PRL_GRAPH_CLIQUE_HPP

#include <boost/graph/graph_traits.hpp>
//#include <boost/graph/graph_utility.hpp>
#include <boost/range/functions.hpp>

#include <prl/stl_concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  namespace dispatch {

    /**
     * Makes two vertices in a graph adjacent if they are not already
     * adjacent.  If the nodes are already adjacent, this method does
     * nothing.
     */
    template <typename G>
    void make_adjacent(typename boost::graph_traits<G>::vertex_descriptor u,
		       typename boost::graph_traits<G>::vertex_descriptor v,
                       G& g,
                       boost::directed_tag,
                       boost::allow_parallel_edge_tag) {
//       if (!boost::is_adjacent(g, u, v))
//         g.add_edge(u, v);
//       if (!boost::is_adjacent(g, v, u))
//         g.add_edge(v, u);

      if (!g.contains(u, v)) g.add_edge(u, v);
      if (!g.contains(v, u)) g.add_edge(v, u);

    }

    /**
     * Makes two vertices in a graph adjacent if they are not already
     * adjacent.  If the nodes are already adjacent, this method does
     * nothing.
     */
    template <typename G>
    void make_adjacent(typename boost::graph_traits<G>::vertex_descriptor u,
		       typename boost::graph_traits<G>::vertex_descriptor v,
                       G& g,
                       boost::directed_tag,
                       boost::disallow_parallel_edge_tag) {
      g.add_edge(u, v);
      g.add_edge(v, u);
    }

    /**
     * Makes two vertices in a graph adjacent if they are not already
     * adjacent.  If the nodes are already adjacent, this method does
     * nothing.
     */
    template <typename G>
    void make_adjacent(typename boost::graph_traits<G>::vertex_descriptor u,
		       typename boost::graph_traits<G>::vertex_descriptor v,
                       G& g,
                       boost::undirected_tag,
                       boost::allow_parallel_edge_tag) {
      if (!g.contains(u,v));
        g.add_edge(u, v);
    }

    /**
     * Makes two vertices in a graph adjacent if they are not already
     * adjacent.  If the nodes are already adjacent, this method does
     * nothing.
     */
    template <typename G>
    void make_adjacent(typename boost::graph_traits<G>::vertex_descriptor u,
		       typename boost::graph_traits<G>::vertex_descriptor v,
                       G& g,
                       boost::undirected_tag,
                       boost::disallow_parallel_edge_tag) {
      g.add_edge(u, v);
    }

  } // namespace prl::dispatch

  //! \addtogroup graph_algorithms
  //! @{

  /**
   * Makes two vertices in a graph adjacent if they are not already
   * adjacent.  If the nodes are already adjacent, this method does
   * nothing.
   */
  template <typename G>
  void make_adjacent(typename boost::graph_traits<G>::vertex_descriptor u,
		     typename boost::graph_traits<G>::vertex_descriptor v,
		     G& g) {
    //concept_assert((EdgeMutableGraph<G>));
    typedef boost::graph_traits<G> traits;
    dispatch::make_adjacent(u, v, g,
                            typename traits::directed_category(),
                            typename traits::edge_parallel_category());
  }

  /**
   * Adds an undirected edge in a directed representation of a graph.
   */
  template <typename G>
  void add_undir_edge(typename boost::graph_traits<G>::vertex_descriptor u,
                      typename boost::graph_traits<G>::vertex_descriptor v,
                      G& g) {
    //concept_assert((EdgeMutableGraph<G>));
    add_edge(u, v, g);
    add_edge(v, u, g);
  }

  /**
   * Places a clique of edges over a set of vertices.  The graph type
   * must implement the Boost AdjacencyGraph and MutableGraph
   * concepts.
   *
   * @param begin a forward iterator over vertex descriptors
   * @param end   a forward iterator over vertex descriptors
   * @param g
   *        the graph that is modified by placing a clique of edges
   *        over the vertices enumerated by [begin, end)
   */
  template <typename It, typename G>
  void make_clique(const It begin, const It end, G& g) {
    concept_assert((InputIterator<It>));
    //concept_assert((EdgeMutableGraph<G>));

    It cur = begin;
    for (cur = begin; cur != end; ++cur) {
      It next = cur;
      while (++next != end)
        make_adjacent(*cur, *next, g);
    }
  }

  template <typename R, typename G>
  void make_clique(const R& vertices, G& g) {
    concept_assert((ReadableForwardRange<R>));
    make_clique(boost::begin(vertices), boost::end(vertices), g);
    // TODO fixme
  }

  //! @}

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
