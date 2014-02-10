
#ifndef SILL_BAYESIAN_GRAPH_HPP
#define SILL_BAYESIAN_GRAPH_HPP

#include <iterator>
#include <set>
#include <map>

#include <sill/graph/directed_graph.hpp>
#include <sill/graph/ancestors_descendants.hpp>
#include <sill/model/markov_graph.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Graph for a Bayesian network, 
   * i.e. DAG with variables associated with nodes.
   *
   * \ingroup model
   */
  template <typename Node,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class bayesian_graph
    : public directed_graph<Node, VertexProperty, EdgeProperty> {

    // Public type declarations
    // =========================================================================
  public:

    //! The type of variables used in this graph
    typedef Node node_type;

    //! The type of domain used by this graph
    typedef std::set<Node> node_set;

    //! The type of edges used in this graph
    typedef directed_edge<Node> edge;

    //! The base type.
    typedef directed_graph<Node, VertexProperty, EdgeProperty> base;

    using base::vertices;
    using base::add_edge;
    using base::add_vertex;

    // Constructors and basic functions
    //==========================================================================
  public:
    //! Default constructor; creates an empty Bayes net graph
    bayesian_graph() {}

    //! Constructs a Bayes net graph with the given set of nodes and no edges.
    bayesian_graph(const node_set& nodes) { 
      foreach(Node v, nodes) {
	add_vertex(v);
      }
    }

    //! Constructs a Bayes net graph with the given set of nodes and no edges.
    bayesian_graph(const forward_range<Node>& nodes) { 
      foreach(Node v, nodes) {
	add_vertex(v);
      }
    }

    //! Creates a Bayes net graph with the given structure.
    //! The graph must be directed
    bayesian_graph(const bayesian_graph<Node>& g) {
      add_nodes(g.nodes());
      foreach(edge e, g.edges())
        add_edge(e.source(), e.target());
    }

    // Queries
    //==========================================================================

    //! Returns the set of variables (of cardinality 1) associated with a vertex
    node_set nodes(Node v) const {
      return make_domain(v);
    }

    //! Returns the variables incident to an edge
    node_set nodes(edge e) const { 
      return make_domain(e.source(), e.target());
    }

    //! Returns the vertices of the graph as a set.
    node_set nodes() const {
      return node_set(boost::begin(vertices()), boost::end(vertices()));
    }

    //! Returns the Markov blanket of a node.
    node_set markov_blanket(Node v) const {
      assert(false); // TODO
      return node_set();
    }

    //! Returns the ancestors of a set of node
    node_set ancestors(const node_set& nodes) const {
      return sill::ancestors(nodes, *this);
    }

    //! Returns the descendants of a set of nodes
    node_set descendants(const node_set& nodes) const {
      return sill::descendants(nodes, *this);
    }

    //! d-separation test
    bool d_separated(const node_set& x, const node_set& y,
                     const node_set& z = node_set()) const { 
      assert(false); 
      return false;
    }

    // Mutators
    //==========================================================================

    //! Adds a node (equivalent to add_vertex(v))
    void add_node(Node v, const VertexProperty& p = VertexProperty()) { 
      base::add_vertex(v, p); 
    }

    //! Adds a collection of nodes
    void add_nodes(const node_set& nodes, bool allow_duplicates = false) { 
      foreach(Node v, nodes) {
        // TODO: should not this be be if (allow_duplicates ...)?
	if (!allow_duplicates || !base::contains(v))
          base::add_vertex(v);
      }
    }

    //! Removes a node
    void remove_node(Node v) {
      base::remove_vertex(v);
    }

    //! Adds all edges from vars to v; add nodes as necessary.
    void add_family(const node_set& vars, Node v) {
      assert(!(vars.count(v)));
      node_set tempset = vars; tempset.insert(v);
      add_nodes(tempset, true);
      foreach(Node u, vars)
        add_edge(u, v);
    }

  }; // class bayesian_graph


  /**
   * Create a Markov graph from a Bayes net graph.
   * \todo Turn into a conversion constructor or a conversion operator
   */
  template <typename Node, typename VertexProperty, typename EdgeProperty>
  markov_graph<Node>
  bayes2markov_graph(const bayesian_graph<Node,VertexProperty,EdgeProperty>& bg) {
    markov_graph<Node> mg(bg.nodes());
    foreach(Node v, bg.vertices()) {
      std::set<Node> clique(bg.parents(v).first, bg.parents(v).second); 
      clique.insert(v);
      mg.add_clique(clique);
    }
    return mg;
  }

}  // namespace sill

namespace boost {

  //! A traits class that lets bayesian_graph work in BGL algorithms
  template <typename Node, typename VP, typename EP>
  struct graph_traits< sill::bayesian_graph<Node, VP, EP> >
    : public graph_traits< sill::directed_graph<Node, VP, EP> > { };

} // namespace boost

#include <sill/macros_undef.hpp>

#endif // SILL_BAYESIAN_GRAPH_HPP
