
#ifndef PRL_CLUSTER_GRAPH_HPP
#define PRL_CLUSTER_GRAPH_HPP

#include <sstream>
#include <vector>
#include <iterator>
#include <set>
#include <boost/bind.hpp>

#include <prl/global.hpp>
#include <prl/range/algorithm.hpp>
#include <prl/datastructure/set_index.hpp>
#include <prl/graph/undirected_graph.hpp>
#include <prl/graph/tree_traversal.hpp>
#include <prl/graph/connected.hpp>
#include <prl/graph/test_tree.hpp>
#include <prl/graph/subgraph.hpp>

#include <prl/range/concepts.hpp>
#include <prl/range/transformed.hpp>
#include <prl/range/forward_range.hpp>
#include <prl/range/io.hpp>

#include <prl/stl_concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  namespace impl {

    /**
     * The information stored with each vertex of the cluster graph.
     */
    template <typename Node, typename VertexProperty>
    struct cg_vertex_info {

      //! The cluster of the primary graph associated with this vertex.
      std::set<Node> cluster;

      //! The externally managed information.
      VertexProperty property;

      //! Default constructor. Default-initializes the property
      //! (in case it is a POD type).
      cg_vertex_info() : property() { }

      //! Standard constructor
      cg_vertex_info(const std::set<Node>& cluster,
                     const VertexProperty& property = VertexProperty())
        : cluster(cluster), property(property) { }
    };

    template <typename Node, typename VP>
    std::ostream& operator<<(std::ostream& out,
                             const cg_vertex_info<Node, VP>& info) {
      out << "(" << info.cluster << " " << info.property << ")";
      return out;
    }

    /**
     * The information stored with each edge of the cluster graph.
     */
    template <typename Node, typename EdgeProperty>
    struct cg_edge_info {

      //! The intersection of the two adjacent clusters.
      std::set<Node> separator;

      //! The externally managed information.
      EdgeProperty property;

      //! Let {u, v} with u < v be the edge associated with this info.
      //! These variables temporarily store the variables in the subtree
      //! rooted at u, away from v and vice versa.
      mutable std::set<Node> forward_reachable, reverse_reachable;

      //! Default constructor
      cg_edge_info() : property() { }

      //! Constructor
      cg_edge_info(const std::set<Node>& separator,
                   const EdgeProperty& property = EdgeProperty())
        : separator(separator), property(property) { }
    };

    template <typename Node, typename EP>
    std::ostream& operator<<(std::ostream& out,
                             const cg_edge_info<Node, EP>& info) {
      out << "(" << info.separator << " " << info.property << ")";
      return out;
    }

  } // namespace prl::impl


  /**
   * Represents a cluster graph \f$T\f$. Each vertex and edge of the graph
   * is associated with a set of nodes. We denote these sets a cluster and
   * separator, respectively. The graph is undirected. The nodes and edges
   * can be associated with custom user properties.
   *
   * \param Node
   *        The element type, stored in clusters and separators.
   *        The type must be DefaultConstructible,CopyConstructible,Comparable.
   * \param VertexProperty
   *        The user information, associated with vertices of the tree.
   *        The type must be DefaultConstructible.
   *
   * This graph can be used in the Boost Graph Library algorithms.
   *
   * \ingroup model
   */
  template <typename Node,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class cluster_graph {
    concept_assert((DefaultConstructible<Node>));
    concept_assert((CopyConstructible<Node>));
    concept_assert((Comparable<Node>));
    concept_assert((DefaultConstructible<VertexProperty>));
    concept_assert((DefaultConstructible<EdgeProperty>));

    // Private type declarations and data members
    // =========================================================================
  private:
    //! information associated with each vertex
    typedef impl::cg_vertex_info<Node, VertexProperty> vertex_info;

    //! information associated with each (undirected) edge
    typedef impl::cg_edge_info<Node, EdgeProperty> edge_info;

    //! An index that maps variables (nodes) to vertices this cluster graph.
    set_index<std::set<Node>, size_t> cluster_index;

    //! The underlying graph
    undirected_graph<size_t, vertex_info, edge_info> graph;

    //! The next vertex id
    size_t next_vertex;


    // Public type declarations
    // =========================================================================
  public:

    //! the underlying graph type (needs to be public for SWIG)
    typedef undirected_graph<size_t, vertex_info, edge_info> graph_type;

    // Graph types
    // (we use the specific types here, so that we do not have manually
    //  instantiate the graph_type template in SWIG)
    typedef size_t vertex;
    typedef undirected_edge<size_t> edge;
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

    //! Constructs an empty cluster graph with no clusters.
    cluster_graph() { }

    //! Swaps two cluster graphs in-place
    void swap(cluster_graph& cg) {
      graph.swap(cg.graph);
      cluster_index.swap(cg.cluster_index);
      std::swap(next_vertex, cg.next_vertex);
    }

    //! Prints a human-readable representation of the cluster graph to
    //! the supplied output stream.
    template <typename OutputStream>
    void print(OutputStream& out) const {
      out << graph;
    }

    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

    // Graph accessors
    // =========================================================================

    //! Returns an ordered set of all vertices
    std::pair<vertex_iterator, vertex_iterator>
    vertices() const {
      return graph.vertices();
    }

    //! Returns the vertices adjacent to u
    std::pair<neighbor_iterator, neighbor_iterator>
    neighbors(const vertex& u) const {
      return graph.neighbors(u);
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

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u)
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

    //! Returns the clusters of this junction tree
    forward_range<const node_set&> clusters() const {
      return make_transformed(vertices(), cluster_functor(this));
    }

    //! Returns the union of all clusters
    node_set nodes() const {
      node_set result;
      foreach(vertex v, vertices()) {
        result = set_union(result, cluster(v));
      }
      return result;
    }

    // Move / copy the following functions from junction_tree?
    // find_cluster_cover
    // find_cluster_meets
    // find_separator_cover
    // find_intersecting_clusters

    //! Returns true if the graph is connected
    bool connected() const {
      return prl::is_connected(graph);
    }

    //! Returns true if the cluster graph is a connected tree
    bool tree() const {
      return num_edges() == num_vertices() - 1 && prl::is_connected(graph);
    }

    //! Returns true if the cluster graph satisfies
    //! the running intersection property. \todo Describe the property
    bool running_intersection() const {
      if (empty()) return true;
      foreach(node_type node, nodes()) {
        size_t n = cluster_index.count(node);
        vertex v = cluster_index[node];
        // Check if the the edges form a spanning tree
        size_t nreachable = test_tree(*this, v, edge_filter(node));
        if (nreachable != n) return false;
      }
      return true;
    }

    //! Returns true if the cluster graph represents a triangulated model
    //! (i.e., is a tree and satisfies the running intersection property).
    bool triangulated() const {
      return tree() && running_intersection();
    }

    /**
     * Returns a subtree, starting from a given vertex and a certain
     * number of hops away.
     */
    cluster_graph subgraph(vertex root, size_t nhops) const {
      cluster_graph new_cg;
      prl::subgraph(graph, root, nhops, new_cg.graph);
      foreach(vertex v, new_cg.vertices())
        new_cg.cluster_index.insert(cluster(v), v);
      new_cg.next_vertex = next_vertex;
      return new_cg;
    }

    // Mutating operations
    //==========================================================================

    /**
     * Adds a vertex with the given cluster and vertex property.
     * If the vertex already exists, replaces the property.
     * \todo What about the edges, separators. etc.
     * \return true if the vertex already exists
     */
    bool add_cluster(vertex v,
                     const node_set& cluster,
                     const VertexProperty& vp = VertexProperty()) {
      assert(v);
      next_vertex = std::max(next_vertex, v + 1);
      bool is_present = graph.add_vertex(v, vertex_info(cluster, vp));
      if (is_present)
        set_cluster(v, cluster);
      else
        cluster_index.insert(cluster, v);
      return is_present;
    }

    /**
     * Adds an edge to the graph.
     * If the edge already exists, overwrites the data.
     */
    std::pair<edge, bool>
    add_separator(vertex u, vertex v,
                  const node_set& separator,
                  const EdgeProperty& ep = EdgeProperty()) {
      assert(separator.subset_of(cluster(u)));
      assert(separator.subset_of(cluster(v)));
      return graph.add_edge(u, v, edge_info(separator, ep));
    }

    /**
     * Adds an edge to the graph, setting the separator to
     * intersection of the two clusters at the endpoints.
     */
    std::pair<edge, bool>
    add_edge(vertex u, vertex v) {
      return graph.add_edge(u, v, set_intersect(cluster(u), cluster(v)));
    }

    //! Sets the cluster associated with an existing vertex.
    //! \todo What are we supposed to do with the separators?
    void set_cluster(vertex v, const node_set& cluster) {
      assert(false); // not implemented properly yet
      /*
      graph[v].cluster = cluster;
      node_set old_cluster = info(v).cluster;
      if (new_cluster == old_cluster) return;

      cluster_index.remove(old_cluster, v);
      info(v).cluster = new_cluster;
      cluster_index.insert(new_cluster, v);

      // Update all incident separators.
      foreach(edge_descriptor e, out_edges(v))
        info(e).separator = new_cluster.intersect(cluster(target(e)));
      */
    }

    //! Sets the separator associated with an edge.
    void set_separator(edge e, const node_set& separator) {
      assert(separator.subset_of(cluster(e.source())));
      assert(separator.subset_of(cluster(e.target())));
      graph[e].separator = separator;
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

    //! Removes all vertices and edges from the graph
    void clear() {
      graph.clear();
      cluster_index.clear();
    }

    // Private member classes
    //==========================================================================

    /**
     * A functor that given a vertex, returns the corresponding cluster.
     */
    class cluster_functor
      : public std::unary_function<vertex, const node_set&> {
      const cluster_graph* cg_ptr;
    public:
      cluster_functor(const cluster_graph* cg_ptr) : cg_ptr(cg_ptr) { }
      const node_set& operator()(vertex v) const { return cg_ptr->cluster(v); }
    };

    /**
     * A functor that returns true when an edge's separator contains a node.
     */
    class edge_filter
      : public std::binary_function<edge, const cluster_graph&, bool> {
      node_type node; //!< the tested node
    public:
      edge_filter(node_type node) : node(node) {}
      bool operator()(edge e, const cluster_graph& g) const {
        return g.separator(e).count(node) > 0;
      }
    };

  }; // class cluster_graph

  /**
   * Prints a human-readable representation of the cluster graph to
   * the supplied output stream.
   * \relates cluster_graph
   */
  template <typename Node, typename VP, typename EP>
  std::ostream& operator<<(std::ostream& out,
                           const cluster_graph<Node, VP, EP>& cg) {
    cg.print(out);
    return out;
  }


} // namespace prl


#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_CLUSTER_GRAPH_HPP




#if 0
    /**
     * A constructor that builds a cluster graph which is a Bethe approximation
     * (i.e. a factor graph).
     */
    template <typename ClusterRange>
      cluster_graph(const ClusterRange& clusters) {
      concept_assert((InputRangeConvertible<ClusterRange, node_set>));
    initialize(clusters);
    }

    /**
     * A constructor that builds a cluster graph which is a Bethe approximation
     * (i.e. a factor graph), with the associated vertex properties for the
     * cluster clusters.
     *
     * @param clusters          Clusters to add to the cluster graph.
     * @param properties       Corresponding properties of the clusters.
     * @param default_property Function for creating default property value for
     *                         additional clusters which this constructor
     *                         creates.  Must take the clusters and return the
     *                         property.
     *
     * \todo Add concept_assert for properties and default_property.
     */
    template <typename ClusterRange, typename InputIterator,
              typename DefaultPropertyConstructor>
    cluster_graph(const ClusterRange& clusters, InputIterator properties,
                  DefaultPropertyConstructor default_property
                  = default_property_functor()) {
      concept_assert((InputRangeConvertible<ClusterRange, node_set>));
      initialize(clusters, properties, default_property);
    }


   /**
     * This method initializes the edge structure of the cluster graph
     * and the separators associated with each edge.
     * It does a Bethe approximation (i.e. creates a factor graph):
     *  - Keep all existing clusters (the clusters)
     *  - Create a new cluster for every variable
     *  - Connect each factor with the variables in its domain
     * Note that if some clusters are single-variable clusters, there will be
     * multiple clusters for that single variable.
     */
    virtual void initialize_edges() {
      if (empty()) return;

      // Create a cluster for each variable.
      node_set variables;
      set<vertex_descriptor> cluster_vertices(vertices());
      foreach(vertex_descriptor v, cluster_vertices)
        variables.insert(cluster(v));
      std::map<node_type, vertex_descriptor> n2v;
      foreach(node_type n, variables)
        n2v[n] = add_vertex(node_set(n));

      // For each cluster vertex,
      //  create edges between it and the nodes for its variables.
      foreach(vertex_descriptor v, cluster_vertices) {
        foreach(node_type n, cluster(v))
          add_edge(v, n2v[n]);
      }
    }

    /**
     * This method initializes the edge structure of the cluster graph
     * and the separators associated with each edge.
     * It does a Bethe approximation (i.e. creates a factor graph):
     *  - Keep all existing clusters (the clusters)
     *  - Create a new cluster for every variable
     *  - Connect each factor with the variables in its domain
     * Note that if some clusters are single-variable clusters, there will be
     * multiple clusters for that single variable.
     */
    template <typename DefaultPropertyConstructor>
    void initialize_edges(DefaultPropertyConstructor default_property) {
      if (empty()) return;

      // Create a cluster for each variable.
      node_set variables;
      set<vertex_descriptor> cluster_vertices(vertices());
      foreach(vertex_descriptor v, cluster_vertices)
        variables.insert(cluster(v));
      std::map<node_type, vertex_descriptor> n2v;
      foreach(node_type n, variables)
        n2v[n] = add_vertex(node_set(n), default_property(node_set(n)));

      // For each cluster vertex,
      //  create edges between it and the nodes for its variables.
      foreach(vertex_descriptor v, cluster_vertices) {
        foreach(node_type n, cluster(v))
          add_edge(v, n2v[n]);
      }
    }



    /**
     * A functor for creating a default vertex property.
     * Used in cluster_graph constructor.
     */
    struct default_property_functor {
      VertexProperty operator()(node_set n) {
        return VertexProperty();
      }
    };


    /**
     * Initializes the cluster graph from a collection of triangulated clusters.
     */
    template <typename ClusterRange>
    void initialize(const ClusterRange& clusters) {
      concept_assert((InputRangeConvertible<ClusterRange, node_set>));
      clear();
      foreach(const node_set& cluster, clusters) add_vertex(cluster);
      initialize_edges();
    }

    /**
     * Initializes the cluster graph from a collection of triangulated clusters.
     * The corresponding vertex properties are specified by properties, and
     * default_property is used to initialize properties for edges.
     */
    template <typename ClusterRange, typename InputIterator,
              typename DefaultPropertyConstructor>
    void initialize(const ClusterRange& clusters, InputIterator properties,
                    DefaultPropertyConstructor default_property
                    = default_property_functor()) {
      concept_assert((InputRangeConvertible<ClusterRange, node_set>));
      clear();
      foreach(const node_set& cluster, clusters)
        add_vertex(cluster, *properties++);
      initialize_edges(default_property);
    }


#endif // #if 0
