#ifndef PRL_REGION_GRAPH_HPP
#define PRL_REGION_GRAPH_HPP

#include <algorithm> // for std::max
#include <set>

#include <prl/datastructure/set_index.hpp>
#include <prl/graph/directed_graph.hpp>
#include <prl/graph/property_functors.hpp>
#include <prl/graph/ancestors_descendants.hpp>
#include <prl/graph/graph_traversal.hpp>

#include <prl/range/forward_range.hpp>
#include <prl/range/transformed.hpp>

// #include <prl/stl_io.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  namespace impl {

    /**
     * The information stored with each vertex of the cluster graph.
     */
    template <typename Node, typename VertexProperty>
    struct rg_vertex_info {

      //! The cluster of the primary graph associated with this vertex.
      std::set<Node> cluster;

      //! The counting number
      int counting_number;

      //! The externally managed information.
      VertexProperty property;

      //! Default constructor. Default-initializes the property
      //! (in case it is a POD type).
      rg_vertex_info() : counting_number(), property() { }

      //! Standard constructor
      explicit rg_vertex_info(const std::set<Node>& cluster,
                              const VertexProperty& property = VertexProperty())
        : cluster(cluster), counting_number(), property(property) { }
    };

    template <typename Node, typename VP>
    std::ostream& operator<<(std::ostream& out,
                             const rg_vertex_info<Node, VP>& info) {
      out << "(" << info.cluster 
          << " " << info.counting_number
          << " " << info.property
          << ")";
      return out;
    }

    /**
     * The information stored with each edge of the cluster graph.
     */
    template <typename Node, typename EdgeProperty>
    struct rg_edge_info {

      //! The intersection of the two adjacent clusters.
      std::set<Node> separator;

      //! The externally managed information.
      EdgeProperty property;

      //! Default constructor
      rg_edge_info() : property() { }

      //! Constructor
      explicit rg_edge_info(const std::set<Node>& separator,
                            const EdgeProperty& property = EdgeProperty())
        : separator(separator), property(property) { }
    };

    template <typename Node, typename EP>
    std::ostream& operator<<(std::ostream& out,
                             const rg_edge_info<Node, EP>& info) {
      out << "(" << info.separator << " " << info.property << ")";
      return out;
    }

  } // namespace prl::impl

  /**
   * This class represents a region grxc vmmmnm aph, see Yedidia 2005.
   * Each vertex in the graph (a size_t) is associated with a region
   * and a counting number.
   *
   * @tparam Node a type that satisfies the Vertex concept.
   * @tparam VertexProperty the property associated with a vertex.
   *         Models the DefaultConstructible and CopyConstructible concepts.
   * @tparam EdgeProperty the property associated with an edge.
   *         Models the DefaultConstructible and CopyConstructible concepts.
   *
   * \ingroup model
   */
  template <typename Node,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class region_graph {
    concept_assert((DefaultConstructible<VertexProperty>));
    concept_assert((DefaultConstructible<EdgeProperty>));
    concept_assert((CopyConstructible<VertexProperty>));
    concept_assert((CopyConstructible<EdgeProperty>));
    
    // Private type declarations and data members
    // =========================================================================
  private:
    //! information associated with each vertex
    typedef impl::rg_vertex_info<Node, VertexProperty> vertex_info;

    //! information associated with each (undirected) edge
    typedef impl::rg_edge_info<Node, EdgeProperty> edge_info;

    //! An index that maps variables (nodes) to vertices this region graph.
    set_index<std::set<Node>, size_t> cluster_index;

    //! The underlying graph
    directed_graph<size_t, vertex_info, edge_info> graph;

    //! The next vertex id
    size_t next_vertex;
    
    // Public type declarations
    // =========================================================================
  public:

    //! the underlying graph type (needs to be public for SWIG)
    typedef directed_graph<size_t, vertex_info, edge_info> graph_type;
    
    // Graph types
    // (we use the specific types here, so that we do not have manually
    //  instantiate the graph_type template in SWIG)
    typedef size_t vertex;
    typedef directed_edge<size_t> edge;
    typedef VertexProperty vertex_property;
    typedef EdgeProperty edge_property;

    // Graph iterators
    typedef typename graph_type::vertex_iterator    vertex_iterator;
    typedef typename graph_type::neighbor_iterator  neighbor_iterator;
    typedef typename graph_type::edge_iterator      edge_iterator;
    typedef typename graph_type::in_edge_iterator   in_edge_iterator;
    typedef typename graph_type::out_edge_iterator  out_edge_iterator;

    // Model-specific types
    //! The type of nodes stored in the graph.
    typedef Node node_type;

    //! The type used to represent clusters and separators.
    typedef std::set<Node> node_set;

    // Constructors and basic member functions
    // =========================================================================
  public:

    //! Constructs an empty region graph with no clusters.
    region_graph() : next_vertex(1) { }

    //! Swaps two region graphs in-place
    void swap(region_graph& rg) {
      graph.swap(rg.graph);
      cluster_index.swap(rg.cluster_index);
      std::swap(next_vertex, rg.next_vertex);
    }

    //! Prints a human-readable representation of the region graph to
    //! the supplied output stream.
    void print(std::ostream& out) const {
      out << graph;
    }

    // Graph accessors
    // =========================================================================

    //! Returns an ordered set of all vertices
    std::pair<vertex_iterator, vertex_iterator>
    vertices() const {
      return graph.vertices();
    }

    //! Returns the parents of u
    std::pair<neighbor_iterator, neighbor_iterator>
    parents(const vertex& u) const {
      return graph.parents(u);
    }

    //! Returns the children of u
    std::pair<neighbor_iterator, neighbor_iterator>
    children(const vertex& u) const {
      return graph.children(u);
    }

    //! Returns the ancestors of one or more vertices
    std::set<vertex> ancestors(const std::set<vertex>& vertices) const {
      return prl::ancestors(vertices, *this);
    }

    //! Returns the ancestors of one vertices
    std::set<vertex> ancestors(const vertex v) const {
      std::set<vertex> temp;
      temp.insert(v);
      return ancestors(temp);
    }

    //! Returns the descendants of a vertex
    std::set<vertex> descendants(const std::set<vertex>& vertices) const {
      return prl::descendants(vertices, *this);
    }

    //! Returns the descendants of one vertices
    std::set<vertex> descendants(const vertex v) const {
      std::set<vertex> temp;
      temp.insert(v);
      return descendants(temp);
    }

    //! Returns all edges in the graph
    std::pair<edge_iterator, edge_iterator>
    edges() const {
      return graph.edges();
    }

    //! Returns the edges incoming to a vertex, such that e.target() == u.
    std::pair<in_edge_iterator, in_edge_iterator>
    in_edges(const vertex& u) const {
      return graph.in_edges(u);
    }

    //! Returns the outgoing edges from a vertex, such that e.source() == u.
    std::pair<out_edge_iterator, out_edge_iterator>
    out_edges(const vertex& u) const {
      return graph.out_edges(u);
    }

    //! Returns true if the graph contains the given vertex
    bool contains(const vertex& u) const {
      return graph.contains(u);
    }

    //! Returns true if the graph contains an undirected edge {u, v}
    bool contains(const vertex& u, const vertex& v) const {
      return graph.contains(u, v);
    }

    //! Returns true if the graph contains an undirected edge
    bool contains(const edge& e) const {
      return graph.contains(e);
    }

    //! Returns an undirected edge with e.source()==u and e.target()==v.
    //! The edge must exist.
    edge get_edge(const vertex& u,  const vertex& v) const {
      return graph.get_edge(u, v);
    }

    //! Returns the number of edges adjacent to a vertex
    size_t in_degree(const vertex& u) const {
      return graph.in_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex
    size_t out_degree(const vertex& u) const {
      return graph.out_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex
    size_t degree(const vertex& u) const {
      return graph.degree(u);
    }

    //! Returns true if the graph has no vertices
    bool empty() const {
      return graph.empty();
    }

    //! Returns the number of vertices
    size_t num_vertices() const {
      return graph.num_vertices();
    }

    //! Returns the number of edges
    size_t num_edges() const {
      return graph.num_edges();
    }

    //! Given a directed edge (u, v), returns a directed edge (v, u)
    //! The edge (v, u) must exist.
    edge reverse(const edge& e) const { 
      return graph.reverse(e);
    }

    //! Returns the property associated with a vertex
    vertex_property& operator[](const vertex& u) {
      return graph[u].property;
    }

    //! Returns the property associated with an edge
    edge_property& operator[](const edge& e) {
      return graph[e].property;
    }

    //! Returns the property associated with a vertex
    const vertex_property& operator[](const vertex& u) const {
      return graph[u].property;
    }

    //! Returns the property associated with an edge
    const edge_property& operator[](const edge& e) const {
      return graph[e].property;
    }

    //! Returns a view of all vertex properties.
    //! The junction tree must outlive the returned view.
    forward_range<const vertex_property&> vertex_properties() const {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    //! Returns a view of all edge properties
    //! The junction tree must outlive the returned view.
    forward_range<const edge_property&> edge_properties() const {
      return make_transformed(edges(), edge_property_functor(*this));
    }

    // Queries
    //==========================================================================

    //! Returns the cluster associated with a vertex.
    const node_set& cluster(vertex v) const {
      return graph[v].cluster;
    }

    //! Returns the separator associated with an edge.
    const node_set& separator(edge e) const {
      return graph[e].separator;
    }

    //! Returns the counting number of a region
    int counting_number(vertex v) const {
      return graph[v].counting_number;
    }

    //! Returns the clusters of this junction tree
    forward_range<const node_set&> clusters() const {
      return make_transformed(vertices(), cluster_functor(this));
      // cluster_functor is defined below
    }

    //! Returns the union of all clusters
    node_set nodes() const {
      node_set result;
      foreach(vertex v, vertices()) result += cluster(v);
      return result;
    }

    //! Returns the vertex that covers the given set of nodes
    //! or 0 if none.
    vertex find_cover(const node_set& set) const {
      vertex v = cluster_index.find_min_cover(set);
      assert(!v || includes(cluster(v), set));
      return v;
    }
    
    //! Returns the a root vertex that covers the given set of nodes or if none.
    //! The region graph must be valid.
    vertex find_root_cover(const node_set& set) const {
      vertex v = cluster_index.find_min_cover(set);
      if (v) {
        while (in_degree(v) > 0) 
          v = *parents(v).first; // chose
      }
      return v;
    }

    // Mutating operations
    //==========================================================================

    /**
     * Adds a new region with the given cluster.
     */
    vertex add_region(const node_set& cluster, 
                      const VertexProperty& vp = VertexProperty()) {
      vertex v = next_vertex++;
      graph.add_vertex(v, vertex_info(cluster, vp));
      cluster_index.insert(cluster, v);
      return v;
    }

    /**
     * Adds a region with the given cluster.
     * The region must not previously exist.
     */
    bool add_region(vertex v,
                    const node_set& cluster,
                    const VertexProperty& vp = VertexProperty()) {
      assert(v);
      next_vertex = std::max(next_vertex, v + 1);
      bool is_present = graph.add_vertex(v, vertex_info(cluster, vp));
      assert(!is_present);
      cluster_index.insert(cluster, v);
      return is_present;
    }

    /**
     * Adds an edge to the graph, setting the separator to
     * intersection of the two clusters at the endpoints.
     */
    std::pair<edge, bool>
    add_edge(vertex u, vertex v, const EdgeProperty& ep = EdgeProperty()) {
      return graph.add_edge(u, v, 
                            edge_info(set_intersect(cluster(u), cluster(v)), ep));
    }

    //! Removes a vertex and the associated cluster and user information
    void remove_vertex(vertex v) {
      cluster_index.remove(cluster(v), v);
      graph.remove_vertex(v);
    }

    //! Removes an undirected edge {u, v} and the associated separator and data.
    void remove_edge(vertex u, vertex v) {
      graph.remove_edge(u, v);
    }

    //! Removes all edges incindent to a vertex
    void clear_edges(vertex u) {
      graph.clear_edges(u);
    }

    //! Removes all edges from the graph
    void clear_edges() {
      graph.clear_edges();
    }

    //! Removes all edges incoming to a vertex
    void clear_in_edges(const vertex& u) {
      graph.clear_in_edges();
    }

    //! Removes all edges outgoing from a vertex
    void clear_out_edges(const vertex& u) {
      graph.clear_out_edges();
    }

    //! Removes all vertices and edges from the graph
    void clear() {
      graph.clear();
      cluster_index.clear();
      next_vertex = 1;
    }

    /**
     * Recomputes the counting numbers.
     * Assigns each root a counting number 1, and sets the remaining
     * clusters to satisfy the running intersection property.
     */
    void update_counting_numbers() {
      std::vector<vertex> order = directed_partial_vertex_order(*this);
//       using namespace std;
//       cout << order << endl;

      foreach(vertex v, order) {
        if(in_degree(v) == 0) // root?
          graph[v].counting_number = 1;
        else {
          int sum = 0;
          foreach(vertex u, ancestors(v)) 
            sum += graph[u].counting_number;
          graph[v].counting_number = 1 - sum;
        }
      }
    }
    
    // Public member classes
    //==========================================================================
    /**
     * A functor that given a vertex, returns the corresponding cluster.
     */
    class cluster_functor
      : public std::unary_function<vertex, const node_set&> {
      const region_graph* rg_ptr;
    public:
      cluster_functor(const region_graph* rg_ptr) : rg_ptr(rg_ptr) { }
      const node_set& operator()(vertex v) const { return rg_ptr->cluster(v); }
    };

    /**
     * A functor that compares two regions based on their cluster sizes.
     */
    class cluster_size_less
      : public std::binary_function<vertex, vertex, bool> {
      const region_graph* rg_ptr;
    public:
      cluster_size_less(const region_graph* rg_ptr): rg_ptr(rg_ptr) { }
      bool operator()(vertex u, vertex v) const {
        size_t size1 = rg_ptr->cluster(u).size();
        size_t size2 = rg_ptr->cluster(v).size();
        return (size1 < size2);// || (size1 == size2 && u < v);
      }
    };

  }; // class region_graph    


  /**
   * Prints a human-readable representation of the junction tree to
   * the supplied output stream.
   * \relates junction_tree
   */
  template <typename Node, typename VP, typename EP>
  std::ostream&
  operator<<(std::ostream& out, const region_graph<Node, VP, EP>& rg) {
    rg.print(out);
    return out;
  }

} // namespace prl


namespace boost {

  //! A traits class that lets junction_tree work in BGL algorithms
  //! (inherits from the traits class for the underlying undirected_graph)
  template <typename Node, typename VP, typename EP>
  struct graph_traits< prl::region_graph<Node, VP, EP> >
    : public graph_traits<typename prl::region_graph<Node, VP, EP>::graph_type>
    { };

} // namespace boost

#include <prl/macros_undef.hpp>

#endif
