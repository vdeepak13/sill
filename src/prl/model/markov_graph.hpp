
#ifndef PRL_MARKOV_GRAPH_HPP
#define PRL_MARKOV_GRAPH_HPP

#include <iterator>

#include <boost/type_traits/is_same.hpp>

#include <prl/graph/undirected_graph.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Graph for a Markov model.
   *
   * \todo do we want the synonyms that use the "node" terminology?
   * \ingroup model
   */
  template <typename Node,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class markov_graph
    : public undirected_graph< Node, VertexProperty, EdgeProperty > {

    // Public type declarations
    // =========================================================================
  public:
    //! The type of variables used in this graph
    typedef Node node_type;

    //! The type of domain used by this graph
    typedef std::set<Node> domain_type;
    
    typedef undirected_edge< Node > edge;
    // unfortunately, writing "typedef typename base::edge edge" makes it
    // difficult to properly detect the underlying type in SWIG

    //! The base class
    typedef undirected_graph< Node, VertexProperty, EdgeProperty > base;

    using base::make_clique;

    // Constructors and basic functions
    //==========================================================================
  public:
    //! Creates an empty markov graph
    markov_graph() {}

    //! Creates a Markov graph with the given set of nodes and no edges
    markov_graph(const domain_type& nodes) { 
      foreach(Node v, nodes) base::add_vertex(v);
    }

    //! Creates a Markov graph with the given set of nodes and edges
    //! Each edge is specified as std::pair<Node, Node>
    template <typename EdgeRange>
    markov_graph(const domain_type& nodes, const EdgeRange& edges) {
      typedef std::pair<Node, Node> node_pair;
      concept_assert((InputRangeConvertible<EdgeRange, node_pair>));
      foreach(Node v, nodes) base::add_vertex(v);
      foreach(node_pair p, edges) base::add_edge(p.first, p.second);
    }

    //! The conversion from a graph with different properties
    //! This function does not copy the graph properties
    template <typename VP, typename EP>
    markov_graph(const markov_graph<Node, VP, EP>& g)  {
      // Prevent accidental instantiations when the standard copy constructor
      // ought to be used
      static_assert((!boost::is_same<VertexProperty, VP>::value ||
                     !boost::is_same<EdgeProperty, EP>::value));
      foreach(Node v, g.nodes()) 
	base::add_vertex(v);
      foreach(edge e, g.edges())
        base::add_edge(e.source(), e.target());
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    // Queries
    //==========================================================================

    //! Returns the variables incident to an edge
    domain_type nodes(edge e) const {
      return make_domain(e.source(), e.target());
    }

    //! Returns all nodes of the graph
    domain_type nodes() const {
      return domain_type(boost::begin(base::vertices()), 
                         boost::end(base::vertices()));
    }

    //! d-separation test
    bool d_separated(const domain_type& x, const domain_type& y,
                     const domain_type& z = domain_type::empty_set) const {
      // TODO: implement using a DFS
      assert(false);
      return false;
    }

    // Mutators
    //==========================================================================

    //! Adds a node (equivalent to add_vertex(v))
    void add_node(Node v, const VertexProperty& p = VertexProperty()) {
      base::add_vertex(v, p);
    }

    //! Adds a collection of nodes (do we need the second parameter?)
    void add_nodes(const domain_type& nodes, bool allow_duplicates = false) {
      foreach(Node v, nodes) {
	if (!allow_duplicates || !base::contains(v)) 
          base::add_vertex(v);
      }
    }

    //! Removes a node and the edges incident to it (equivalne to remove_vertex)
    void remove_node(Node v) {
      base::remove_vertex(v);
    }

    //! Adds a clique over a set of variables; adds the nodes as necessary
    void add_clique(const domain_type& clique) {
      add_nodes(clique, true);
      make_clique(clique);
    }

  }; // class markov_graph

} // namespace prl

namespace boost {

  //! A traits class that lets markov_graph work in BGL algorithms
  template <typename Node, typename VP, typename EP>
  struct graph_traits< prl::markov_graph<Node, VP, EP> >
    : public graph_traits< prl::undirected_graph<Node, VP, EP> > { };

} // namespace boost

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_MARKOV_GRAPH_HPP
