
#ifndef PRL_LEARNT_JUNCTION_TREE_HPP
#define PRL_LEARNT_JUNCTION_TREE_HPP

#include <prl/model/junction_tree.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * This is a more mutable version of a junction tree.  It allows unsafe
   * operations and is only meant to be used by learnt_decomposable.
   *
   * The constructors and methods here are identical to those in junction_tree;
   * see junction_tree for documentation.
   */
  template <typename Node,
            typename VertexProperty = void_,
            typename EdgeProperty = void_ >
  class learnt_junction_tree
    : public junction_tree<Node,VertexProperty,EdgeProperty>{

    // Public type declarations
    //==========================================================================
  public:
    typedef junction_tree<Node,VertexProperty,EdgeProperty> base;
    typedef typename base::vertex vertex;
    typedef typename base::edge edge;
    typedef typename base::node_set node_set;

    // Constructors
    //==========================================================================
    learnt_junction_tree() { }
    learnt_junction_tree(const base& jt) : base(jt) { }
    template <typename Graph, typename Strategy>
    learnt_junction_tree(Graph& g, Strategy strategy,
                         typename Graph::vertex* = 0)
      : base(g, strategy) { }
    template <typename CliqueRange>
    learnt_junction_tree(const CliqueRange& cliques,
                  typename CliqueRange::iterator * = 0)
      : base(cliques) { }
  #ifdef SWIG
    learnt_junction_tree(const std::vector<node_set>& cliques);
  #endif
    template <typename CliqueRange, typename InputIterator>
    learnt_junction_tree(const CliqueRange& cliques, InputIterator properties,
                         typename CliqueRange::iterator* = 0)
      : base(cliques, properties) { }
    explicit
    learnt_junction_tree(const cluster_graph<Node, VertexProperty,
                                             EdgeProperty>& cg,
                         bool force = false)
      : base(cg, force) { }

    // Public member functions
    //==========================================================================
    /**
     * Inserts a clique to the junction tree with no edges.
     */
    vertex add_clique(const node_set& clique,
                      const VertexProperty& p = VertexProperty()) {
      return base::add_clique(clique, p);
    }

    /**
     * Changes the clique associated with a vertex and updates the
     * incident separators.
     *
     * This method changes the structure of the junction tree.  It is
     * the caller's responsibility to ensure that the junction tree
     * remains singly connected, and that the running intersection
     * property holds.
     *
     * @param  v the vertex whose clique is updated
     * @param  new_clique the new clique associated with the vertex
     */
    void set_clique(vertex v, const node_set& new_clique) {
      base::set_clique(v, new_clique);
    }

    /**
     * Adds an edge to the junction tree and sets the separator
     */
    edge add_edge(vertex u, vertex v) {
      return base::add_edge(u, v);
    }

    /**
     * Adds an edge to the junction tree and sets the separator
     */
    edge add_edge(vertex u, vertex v, const EdgeProperty& ep) {
      return base::add_edge(u, v, ep);
    }

    //! Removes an edge from the junction tree.
    void remove_edge(edge e) {
      base::remove_edge(e);
    }

    //! Removes edge <v1,v2> from the model.
    void remove_edge(vertex v1, vertex v2) {
      base::remove_edge(v1, v2);
    }

    /**
     * Removes an existing vertex and incident edges from the graph.
     * All descriptors and iterators that do not point to the removed
     * vertex remain valid.
     *
     * This method changes the structure of the junction tree.  It is
     * the caller's responsibility to ensure that the junction tree
     * remains singly connected, and that the running intersection
     * property holds.
     */
    void remove(vertex v) {
      base::remove(v);
    }

  };  // class learnt_junction_tree

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNT_JUNCTION_TREE_HPP
