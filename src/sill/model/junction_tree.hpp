
#ifndef SILL_JUNCTION_TREE_HPP
#define SILL_JUNCTION_TREE_HPP

#include <vector>
#include <iterator>
#include <stdexcept>
#include <set>
//#include <boost/pending/disjoint_sets.hpp>

#include <sill/global.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/datastructure/set_index.hpp>
#include <sill/graph/concepts.hpp>
#include <sill/graph/undirected_graph.hpp>
#include <sill/graph/property_functors.hpp>
#include <sill/graph/tree_traversal.hpp>
#include <sill/graph/triangulation.hpp>
#include <sill/graph/mst.hpp>
#include <sill/graph/subgraph.hpp>
#include <sill/model/markov_graph.hpp>

#include <sill/model/cluster_graph.hpp> //! For the conversion constructor

#include <sill/stl_concepts.hpp>

#include <sill/range/concepts.hpp>
#include <sill/range/transformed.hpp>
#include <sill/range/numeric.hpp>
#include <sill/range/io.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  template <typename F> class shafer_shenoy;


  namespace impl {
    /**
     * The information stored with each vertex of the junction tree.
     */
    template <typename Node, typename VertexProperty>
    struct jt_vertex_info {

      //! The clique of the primary graph associated with this vertex.
      std::set<Node> clique;

      //! The externally managed information.
      VertexProperty property;

      //! True if the vertex has been marked.
      mutable bool marked;

      //! Default constructor. Default-initializes the property
      //! (in case it is a POD type).
      jt_vertex_info() : property(), marked(false) { }

      //! Standard constructor
      jt_vertex_info(const std::set<Node>& clique,
                     const VertexProperty& property = VertexProperty())
        : clique(clique), property(property), marked() { }

      //! Serialize members
      void save(oarchive & ar) const {
        ar << clique << property << marked;
      }

      //! Deserialize members
      void load(iarchive & ar) {
        ar >> clique >> property >> marked;
      }

    }; // struct jt_vertex_info
    
    //! Compares the clique and vertex property stored at two vertices
    template <typename Node, typename VertexProperty>
    bool operator==(const jt_vertex_info<Node, VertexProperty>& a,
                    const jt_vertex_info<Node, VertexProperty>& b) {
      return a.clique == b.clique && a.property == b.property;
    }

    template <typename Node, typename VertexProperty>
    bool operator!=(const jt_vertex_info<Node, VertexProperty>& a,
                    const jt_vertex_info<Node, VertexProperty>& b) {
      return a.clique != b.clique || a.property != b.property;
    }

    template <typename Node, typename VP>
    std::ostream& operator<<(std::ostream& out,
                             const jt_vertex_info<Node, VP>& info) {
      out << "(";
      print_range(out, info.clique, '{', ' ', '}');
      out << " " << info.property;
      out << " " << info.marked << ")";
      return out;
    }

    /**
     * The information stored with each edge of the junction tree.
     */
    template <typename Node, typename EdgeProperty>
    struct jt_edge_info {

      //! The intersection of the two adjacent cliques.
      std::set<Node> separator;

      //! The externally managed information.
      EdgeProperty property;

      //! True if the vertex has been marked.
      mutable bool marked;

      //! Let {u, v} with u < v be the edge associated with this info.
      //! These variables temporarily store the variables in the subtree
      //! rooted at u, away from v and vice versa.
      mutable std::set<Node> forward_reachable, reverse_reachable;

      //! Default constructor
      jt_edge_info() : property(), marked() { }

      //! Constructor
      jt_edge_info(const std::set<Node>& separator,
                   const EdgeProperty& property = EdgeProperty())
        : separator(separator), property(property), marked(marked) { }

      //! Serialize members
      void save(oarchive & ar) const {
        ar << separator << property << marked << forward_reachable
           << reverse_reachable;
      }

      //! Deserialize members
      void load(iarchive & ar) {
        ar >> separator >> property >> marked >> forward_reachable
           >> reverse_reachable;
      }

    }; // struct jt_edge_info

    template <typename Node, typename EdgeProperty>
    bool operator==(const jt_edge_info<Node, EdgeProperty>& a,
                    const jt_edge_info<Node, EdgeProperty>& b) {
      return a.separator == b.separator && a.property == b.property;
    }

    template <typename Node, typename EdgeProperty>
    bool operator!=(const jt_edge_info<Node, EdgeProperty>& a,
                    const jt_edge_info<Node, EdgeProperty>& b) {
      return a.separator != b.separator || a.property != b.property;
    }

    template <typename Node, typename EdgeProperty>
    std::ostream& operator<<(std::ostream& out,
                             const jt_edge_info<Node, EdgeProperty>& info) {
      out << "(" << info.separator << " " << info.property
          << " " << info.marked << ")";
      return out;
    }

  } // namespace sill::impl

  /**
   * Represents a junction tree \f$T\f$ over a set of nodes.
   *
   * Each vertex and each (undirected) edge is associated with a set
   * of variables, called a clique and a separator, respectively.
   * Each vertex and each edge can be also associated with
   * user-specified properties. The tree is undirected but, if needed,
   * one can store information about both direction of the edge, using
   * the bidirectional structure.
   *
   * The graph can be used in the Boost.Graph library.
   *
   * @tparam Node a type that satisfies the Vertex concept.
   * @tparam VertexProperty a type that models the DefaultConstructible
   *         and the CopyConstructible concept.
   * @tparam EdgeProperty a type that models the DefaultConstructible
   *         and the CopyConstructible concept. The edge property
   *         should have a fast default constructor, since we temporarily
   *         create a superlinear number of edges to compute the MST.
   *
   * \ingroup model
   */
  template <typename Node,
            typename VertexProperty = void_,
            typename EdgeProperty = void_ >
  class junction_tree {
    concept_assert((DefaultConstructible<VertexProperty>));
    concept_assert((DefaultConstructible<EdgeProperty>));

    template <typename F> friend class decomposable;
    template <typename F> friend class shafer_shenoy; // for set_clique
    template <typename N, typename VP, typename EP> friend class junction_tree;

    // Private type declarations and data members
    // =========================================================================
  private:
    //! information associated with each vertex
    typedef impl::jt_vertex_info<Node, VertexProperty> vertex_info;

    //! information associated with each (undirected) edge
    typedef impl::jt_edge_info<Node, EdgeProperty> edge_info;

    //! An index that maps each variable to the vertices that contain it
    set_index<std::set<Node>, size_t> clique_index;

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

    //! The type used to represent cliques and separators.
    typedef std::set<Node> node_set;

  public:
    // Constructors and basic member functions
    // =========================================================================

    //! Constructs an empty junction tree with no cliques.
    junction_tree() : next_vertex(1) { }

    /*
    //! Constructs a junction tree with the same structure as the specified tree
    junction_tree(const junction_tree<Node>& jt) {
      foreach(vertex v, ...)
    }
    */

    /**
     * A constructor that builds a junction tree for the supplied
     * graph.
     *
     * @param g
     *        The graph for which the junction tree is initialized.
     *        The resulting junction tree will contain all maximal
     *        cliques of the graph.  This operation modifies the graph
     *        to remove all edges.
     * @param vertex_map
     *        A function object that transforms the vertex of graph to
     *        a node (variable) in the junction tree. The map must be 1-1.
     * @param strategy
     *        The elimination strategy used in triangulation.
     *
     * @see min_degree_strategy, min_fill_strategy, identity
     */
    template <typename Graph, typename Strategy>
    junction_tree(Graph& g, Strategy strategy, typename Graph::vertex* = 0)
      : next_vertex(1) {
      // concept_assert((VertexListGraph<Graph>));
      concept_assert((EliminationStrategy<Strategy, Graph>));
      initialize(g, strategy);
    }

    /**
     * A constructor that builds the junction tree from a set of cliques
     * of a triangulated graph.
     * \param remove_nonmaximal if true, removes the non-maximal cliques
     *        from the set.
     */
    template <typename CliqueRange>
    junction_tree(const CliqueRange& cliques,
                  typename CliqueRange::iterator * = 0)
      : next_vertex(1) {
      concept_assert((InputRangeConvertible<CliqueRange, node_set>));
      initialize(cliques);
    }

  #ifdef SWIG
    junction_tree(const std::vector<node_set>& cliques);
  #endif

    /**
     * A constructor that builds the junction tree from the maximal
     * cliques of a triangulated graph, with the associated vertex properties.
     * \param remove_nonmaximal if true, removes the non-maximal cliques
     *        from the set.
     */
    template <typename CliqueRange, typename InputIterator>
    junction_tree(const CliqueRange& cliques, InputIterator properties,
                  typename CliqueRange::iterator* = 0)
      : next_vertex(1) {
      concept_assert((InputRangeConvertible<CliqueRange, node_set>));
      initialize(cliques, properties);
    }

    /**
     * Constructs a junction tree equivalent to the given cluster graph.
     * The graph must be triangulated.
     *
     * \param force
     *        if true, adds variables to the cliques as necessary to
     *        satisfy the running intersection property
     *
     * \throw std::invalid_argument
     *        if force=false and the junction tree is not triangulated.
     *        or if the cluster graph is not a tree
     */
    explicit
    junction_tree(const cluster_graph<Node, VertexProperty, EdgeProperty>& cg,
                  bool force = false) {
      if (!cg.tree())
        throw std::invalid_argument("The cluster graph is not a tree.");

      if (!force && !cg.running_intersection())
        throw std::invalid_argument("The cluster graph does not satisfy the running intersection property.");

      foreach(vertex v, cg.vertices())
        add_clique(v, cg.cluster(v), cg[v]);
      foreach(edge e, cg.edges())
        add_edge(e.source(), e.target(), cg[e]);
      next_vertex = sill::accumulate(vertices(), 0, maximum<size_t>()) + 1;

      if(force) triangulate();
    }

    /**
     * Initializes this junction tree for the supplied graph.  This
     * method assumes this object has unique ownership of the
     * underlying graph.
     *
     * @param graph
     *        The graph for which the junction tree is initialized.
     *        The vertices of the graph must be convertible to the
     *        Node type. The resulting junction tree will contain all
     *        maximal cliques of the graph.  This operation removes
     *        all edges of the graph.
     * @param strategy
     *        The elimination strategy used in triangulation.
     *
     * @see EliminationStrategy
     */
    template <typename Graph, typename Strategy>
    void initialize(Graph& graph, Strategy strategy, typename Graph::vertex* =0)
    {
      //concept_assert((VertexListGraph<Graph>));
      concept_assert((EliminationStrategy<Strategy, Graph>));

      // Use triangulation to compute the cliques.
      std::vector<node_set> graph_cliques;
      sill::triangulate(graph, std::back_inserter(graph_cliques), strategy);
      initialize(graph_cliques);
      // std::cout << graph_cliques << std::endl;
    }

    //! Initializes the junction tree from a collection of triangulated cliques.
    template <typename CliqueRange>
    void initialize(const CliqueRange& cliques,
                    typename CliqueRange::iterator* = 0) {
      concept_assert((InputRangeConvertible<CliqueRange, node_set>));
      clear();
      foreach(node_set clique, cliques) add_clique(clique);
      initialize_edges();
    }

    //! Initializes the junction tree from a collection of triangulated cliques.
    template <typename CliqueRange, typename InputIterator>
    void initialize(const CliqueRange& cliques, InputIterator properties,
                    typename CliqueRange::iterator* = 0) {
      concept_assert((InputRangeConvertible<CliqueRange, node_set>));
      clear();
      foreach(node_set clique, cliques)
        add_clique(clique, *properties++);
      initialize_edges();
    }

    /**
     * Initializes the junction tree to have the same structure and
     * cliques as the specified junction tree.
     *
     * @param jt  Only the nodes (structure) of this tree are used;
     *            any vertex and edge properties are ignored.
     */
    template <typename VertProp, typename EdgeProp>
    void initialize(const junction_tree<Node, VertProp, EdgeProp>& jt) {
      graph.clear();
      clique_index = jt.clique_index;
      next_vertex = jt.next_vertex;
      foreach(vertex v, jt.vertices())
        graph.add_vertex(v, vertex_info(jt.clique(v)));
      foreach(edge e, jt.edges())
        graph.add_edge(e.source(), e.target(), edge_info(jt.separator(e)));
    }

    //! Swaps two junction trees in-place (in constant time).
    void swap(junction_tree& jt) {
      graph.swap(jt.graph);
      clique_index.swap(jt.clique_index);
      std::swap(next_vertex, jt.next_vertex);
    }

    //! Prints a human-readable representation of the junction tree to
    //! the supplied output stream.
    void print(std::ostream& out) const {
      out << graph;
    }

    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << clique_index << graph << next_vertex;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> clique_index >> graph >> next_vertex;
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

    //! Returns a null vertex
    static vertex null_vertex() { return graph_type::null_vertex(); }

    /**
     * Returns true if two junction trees are identical
     * The vertices and vertex / edge properties must match,
     * not only the cliques / separators.
     */
    bool operator==(const junction_tree& other) const {
      return graph == other.graph;
    }
    
    //! Inequality comparison
    bool operator!=(const junction_tree& other) const {
      return graph != other.graph;
    }

    // Queries
    //==========================================================================

    /**
     * Returns a vertex in the tree. The function will return the
     * same vertex, provided that the vertex set does not change.
     * The graph must not be empty.
     */
    vertex root() const {
      assert(!empty());
      return *vertices().first;
    }

    //! Returns the clique associated with a vertex.
    const node_set& clique(vertex v) const {
      return graph[v].clique;
    }

    //! Returns the separator associated with an edge.
    const node_set& separator(edge e) const {
      return graph[e].separator;
    }

    //! Returns the cliques of this junction tree
    forward_range<const node_set&> cliques() const {
      return make_transformed(vertices(), clique_functor(this));
    }

    //! Returns the separators of this junction tree
    forward_range<const node_set&> separators() const {
      return make_transformed(edges(), separator_functor(this));
    }

    //! Returns true if the vertex has been marked.
    bool marked(vertex v) const {
      return graph[v].marked;
    }

    //! Returns true if the edge has been marked.
    bool marked(edge e) const {
      return graph[e].marked;
    }

    //! Returns the union of all cliques
    node_set nodes() const {
      node_set result;
      foreach(vertex v, vertices()) result = set_union(result, clique(v));
      return result;
    }

    //! Returns the maximum clique size minus one.
    int tree_width() const {
      size_t m = 0;
      foreach(vertex v, vertices())
        m = std::max(m, clique(v).size());
      return int(m) - 1;
    }

    /**
     * Returns a vertex whose clique covers (is a superset of) the supplied
     * set of nodes. If there are multiple such vertices,
     * returns the one with the smallest clique size (cardinality).
     * If there is no such vertex, then returns the null vertex.
     */
    vertex find_clique_cover(const node_set& set) const {
      vertex v = clique_index.find_min_cover(set);
      assert(!v || includes(clique(v), set));
      return v;
    }

    /**
     * Returns a vertex whose clique cover meets (intersects) the supplied
     * set of values. The returned vertex is one that has the smallest clique
     * that has maximal intersection with the supplied set.
     */
    vertex find_clique_meets(const node_set& set) const {
      return clique_index.find_max_intersection(set);
    }

    /**
     * Returns an edge whose separator covers the supplied set of
     * variables. If there are multiple such edges, one with the
     * smallest separator is returned.  If there is no such edge,
     * returns none.
     *
     * @param  set the set of nodes for which a separator cover is sought
     */
    edge find_separator_cover(const node_set& set) const {
      typedef std::pair<node_set, vertex> clique_vertex_pair;
      // std::cerr << "Searching for separator cover for " << set << std::endl;
      std::vector<vertex> candidate_vertices;
      clique_index.find_supersets(set, std::back_inserter(candidate_vertices));
      if (candidate_vertices.empty()) return edge();
      edge best;
      size_t size(0);
      foreach(vertex v, candidate_vertices) {
        foreach(edge e, out_edges(v)) {
          if (includes(separator(e),set)) {
            if (best != edge() || separator(e).size() < size) {
              best = e;
              size = separator(e).size();
            }
          }
        }
      }
      return best;
    }

    //! Outputs the vertices whose cliques overlap the supplied set of nodes.
    template <typename OutIt>
    OutIt find_intersecting_cliques(const node_set& set, OutIt out) const {
      concept_assert((OutputIterator<OutIt, vertex>));
      return clique_index.find_intersecting_sets(set, out);
    }

    /**
     * Computes the reachable nodes for each directed edge in the
     * junction tree.  The reachable nodes for an edge (u, v) is the
     * union of all cliques that are at vertices closer to u than v.
     *
     * @param propagate_past_empty
     *        If set to false, then edges with empty separators
     *        are ignored. Their reachable sets are assigned empty sets.
     */
    void compute_reachable(bool propagate_past_empty) const {
      if (empty()) return;
      mpp_traversal(*this, reachable_visitor(propagate_past_empty, NULL));
    }


    /**
     * Computes the reachable nodes for each directed edge in the
     * junction tree.  The reachable nodes for an edge (u, v) is the
     * union of all cliques that are at vertices closer to u than v.
     *
     * @param prop_past_empty
     *        If set to false, then edges with empty separators
     *        are ignored. Their reachable sets are assigned empty sets.
     * @param filter
     *        The reachable node sets are intersected with this set.
     */
    void compute_reachable(bool propagate_past_empty,
                           const node_set& filter) const {
      if (empty()) return;
      mpp_traversal(*this, reachable_visitor(propagate_past_empty, &filter));
    }

    //! Returns a pre-computed reachable set associated with a directed edge
    const node_set& reachable(edge e) const {
      return e.source() < e.target() ?
        graph[e].forward_reachable : graph[e].reverse_reachable;
    }

    /**
     * Marks the smallest subtree (or subforest) of this junction tree
     * whose cliques cover the supplied set of nodes.
     *
     * When the junction tree has empty separators, it can be regarded
     * as a forest of independent junction trees.  In this case, a set
     * of variables may be covered by a non-contiguous subgraph of the
     * total junction tree.  The argument force_continuous controls
     * whether whether the function marks a connected or is allowed to
     * mark a set of disconnected subtree covers.
     *
     * @param set
     *        a set of nodes to be covered
     * @param force_continuous
     *        if true, the function is guaranteed to mark a connected subgraph.
     */
    void mark_subtree_cover(const node_set& set, bool force_continuous) const {
      if (empty()) return;

      // Initialize the vertices to be white.
      foreach(vertex v, vertices()) graph[v].marked = false;

      // Compute the reachable variables for the set.
      compute_reachable(force_continuous, set);

      // The edges that must be in the subtree are those such that the
      // reachable variables in both directions have a non-empty
      // symmetric difference.
      node_set cover;
      foreach(edge e, edges()) {
        vertex u = e.source();
        vertex v = e.target();
        const node_set& reachable1 = graph[e].forward_reachable;
        const node_set& reachable2 = graph[e].reverse_reachable;
        if (! includes(reachable2, reachable1) &&
            ! includes(reachable1, reachable2)) {
          // This edge is present in the subtree.
          graph[e].marked = true;
          graph[u].marked = true;
          graph[v].marked = true;
          cover.insert(clique(u).begin(), clique(u).end());
          cover.insert(clique(v).begin(), clique(v).end());
        } else {
          graph[e].marked = false;
        }
      }

      // We must also mark vertices that are part of the subtree but
      // are not attached to any other vertex in the subtree.
      // If force_continuous = true, then either all nodes were covered
      // in the previous stage, or the nodes are contained in a single clique
      node_set uncovered = set_difference(set, cover);
      while (!uncovered.empty()) {
        vertex v = find_clique_meets(uncovered);
        assert(v);
        assert(set_intersect(clique(v),uncovered).size() > 0);
        foreach(const Node &d, clique(v)) {
          uncovered.erase(d);
        }
        graph[v].marked = true;
      }
    }

    /**
     * Checks that this junction tree is valid.  This method will
     * generate an assertion violation if the junction tree is not
     * valid.  The graph must be singly-connected, its separators must
     * be the intersections of the adjoined cliques, and it must have
     * the running intersection property.
     *
     * This method updates the reachable variables on each edge.
     */
    void check_validity() const {
      // Check that the separators are correct.
      foreach(edge e, edges()) {
        vertex s = e.source();
        vertex t = e.target();
        if (!set_equal(separator(e), 
                       set_intersect(clique(s), clique(t)))) {
          std::cerr << "check_validity() failed: "
                    << "separator(e) != clique(s).intersect(clique(t)):"
                    << std::endl;
          std::cerr << "  separator(e) = " << separator(e)
                    << ", clique(s) = " << clique(s)
                    << ", clique(t) = " << clique(t) << std::endl;
          print(std::cerr);
          assert(false);
        }
      }

      // Check that the running intersection property holds.
      compute_reachable(true);
      foreach(vertex v, vertices()) {
        in_edge_iterator it1, end;
        for (boost::tie(it1, end)=in_edges(v); it1!=end; ++it1) {
          in_edge_iterator it2 = it1;
          while (++it2!=end) {
            assert(includes(clique(v),
                            set_intersect(reachable(*it1), reachable(*it2))));
          }
        }
      }
    }

    /**
     * Returns a subtree, starting from a given vertex and a certain
     * number of hops away.
     */
    junction_tree subtree(vertex root, size_t nhops) const {
      junction_tree new_jt;
      sill::subgraph(graph, root, nhops, new_jt.graph);
      foreach(vertex v, vertices())
        new_jt.clique_index.insert(clique(v), v);
      new_jt.next_vertex = next_vertex;
      return new_jt;
    }

    // operations specific to probabilistic models -----------------------------

    /**
     * Returns true if nodes x are independent from nodes y
     * given nodes z in a distribution that satisfies the
     * independence assumptions captured by this junction tree.
     */
    bool d_separated(const node_set& x, const node_set& y,
                     const node_set& z = node_set::empty_set) const {
      assert(false); // not implemented yet
      return false;
    }

    /**
     * Returns an equivalent markov graph (without the side information).
     */
    sill::markov_graph<Node> markov_graph() const {
      sill::markov_graph<Node> mg(nodes());
      foreach(vertex v, vertices())
        mg.add_clique(clique(v));
      return mg;
    }

    // Mutating operations
    //==========================================================================

    //! Removes all vertices and edges from the graph
    void clear() {
      graph.clear();
      clique_index.clear();
    }

    //! Removes a vertex from the tree. The vertex must be a leaf or a root.
    void remove_vertex(const vertex& u) {
      if (degree(u) <= 1) remove(u);
      else throw std::logic_error("The removed vertex must be a leaf.");
    }

    /**
     * Merges two adjacent vertices of the junction tree. The edge u
     * -- v and the source vertex u are deleted, and the target vertex
     * v is made adjacent to all neighbors of u.  (The information on
     * these edges is copied from the original edges.)  The new clique
     * of v is set as the union of the original cliques of u and v,
     * and the separators remain unchanged; this guarantees the
     * running intersection property still holds.
     *
     * @param  edge the edge whose incident vertices are to be merged
     * @return the retained vertex
     */
    vertex merge(edge e) {
      vertex u = e.source();
      vertex v = e.target();

      // Attach all neighbors of u, except for v, to vertex v.
      foreach(edge eu, in_edges(u)) {
        vertex w = eu.source();
        if (w != v) graph.add_edge(w, v, graph[e]);
      }
      set_clique(v, set_union(clique(u), clique(v)));
      // TODO: don't recompute the separators    FIXME??
      remove(u);
      return v;
    }

    /**
     * Extends the cliques, so that tree satisfies the running intersection
     * property.
     */
    void triangulate() {
      compute_reachable(true);
      foreach(vertex v, vertices()) {
        node_set c = clique(v);
        in_edge_iterator it1, end;
        for (boost::tie(it1, end)=in_edges(v); it1!=end; ++it1) {
          in_edge_iterator it2 = it1;
          while (++it2!=end)
            c.insert(reachable(*it1).intersect(reachable(*it2)));
        }
        set_clique(v, c);
      }
    }

  protected:
    // Protected member functions
    //==========================================================================

    /**
     * This method initializes the edge structure of the junction tree
     * graph and the separators associated with each edge.
     * The graph must have no edges.
     */
    void initialize_edges() {
      assert(graph.num_edges() == 0);
      typedef std::pair<node_set, vertex> clique_vertex_pair;

      if (empty()) return;

      // Select a distinguished vertex of the tree
      vertex root = *vertices().first;

      // For each pair of overlapping cliques, add a candidate edge to the
      // graph using the set index.
      std::vector<vertex> nbrs;
      foreach(vertex u, vertices()) {
        nbrs.clear();
        clique_index.find_intersecting_sets(clique(u),std::back_inserter(nbrs));
        foreach(vertex v, nbrs)
          if (u < v) graph.add_edge(u, v);
        // Add edges between a distinguished vertex and all other vertices.
        // This ensures that the junction tree is connected,
        // even if the original graph is not.
        if (root != u) graph.add_edge(root, u);
      }

      // Compute the edges of a maximum spanning tree
      std::vector<edge> tree_edges;
      // run Kruskal's algorithm with weights equal to the -separator size
      kruskal_minimum_spanning_tree
        (*this,
         std::back_inserter(tree_edges),
         sepsize_functor(this));
      // bind(&junction_tree::negated_sep_size, this, _1));
      // bind(&X::f, this, _1) converts a unary member function into a functor

      // Remove all edges and add the computed edges
      graph.clear_edges();
      foreach(const edge& e, tree_edges)
        add_edge(e.source(), e.target());
    }

    /**
     * Inserts a clique to the junction tree with no edges.
     */
    void add_clique(vertex v, const node_set& clique,
                    const VertexProperty& p = VertexProperty()) {
      assert(!contains(v));
      graph.add_vertex(v, vertex_info(clique, p));
      clique_index.insert(clique, v);
      if (v > next_vertex) next_vertex = v + 1;
    }

    /**
     * Inserts a clique to the junction tree with no edges.
     */
    vertex add_clique(const node_set& clique,
                      const VertexProperty& p = VertexProperty()) {
      vertex v = next_vertex++;
      graph.add_vertex(v, vertex_info(clique, p));
      clique_index.insert(clique, v);
      return v;
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
      const node_set& old_clique = clique(v);
      if (new_clique == old_clique) return;

      clique_index.remove(old_clique, v);
      graph[v].clique = new_clique;
      clique_index.insert(new_clique, v);

      // Update all incident separators.
      foreach(edge e, out_edges(v))
        graph[e].separator = set_intersect(new_clique, clique(e.target()));
    }

    /**
     * Adds an edge to the junction tree and sets the separator
     */
    edge add_edge(vertex u, vertex v) {
      edge e = graph.add_edge(u, v).first;
      graph[e].separator = set_intersect(clique(u), clique(v));
      return e;
    }

    /**
     * Adds an edge to the junction tree and sets the separator
     */
    edge add_edge(vertex u, vertex v, const EdgeProperty& ep) {
      edge e = add_edge(u, v);
      graph[e].property = ep;
      return e;
    }

    //! Removes an edge from the junction tree.
    void remove_edge(edge e) {
      graph.remove_edge(e.source(), e.target());
    }

    //! Removes edge <v1,v2> from the model.
    void remove_edge(vertex v1, vertex v2) {
      graph.remove_edge(v1, v2);
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
      clique_index.remove(clique(v), v);
      graph.remove_vertex(v);
    }

  private:
    // Private member classes
    //==========================================================================

    /**
     * An edge visitor that computes the reachable nodes along an
     * edge.  This visitor must be used in a message passing protocol
     * (MPP) traversal to work properly.
     */
    class reachable_visitor {
      bool propagate_past_empty;
      const node_set* filter;
    public:
      /**
       * Constructor.
       *
       * @param filter
       *        If not NULL, the reachable set is intersected with *filter.
       *        Use boost::none to specify a null value.
       * @param propagate_past_empty
       *        If this flag is false, then edges with empty
       *        separators are treated as if they were not there.
       *        Their reachable nodes are empty.
       */
      reachable_visitor(bool propagate_past_empty, const node_set* filter)
        : propagate_past_empty(propagate_past_empty), filter(filter) { }

      //! Invoked when the edge e = (u, v) is traversed in the u --> v direction
      void operator()(edge e, const junction_tree& jt) {
        node_set reachable;
        vertex s = e.source();
        vertex t = e.target();

        // If the separator is empty and we are not propagating
        // reachable variables past empty separator (i.e., we are
        // treating empty separators as missing edges), then we're done.
        if (!jt.separator(e).empty() || propagate_past_empty) {
          foreach(edge d, jt.in_edges(s)) {
            if (d.source() != t)
              reachable = set_union(reachable, jt.reachable(d));
          }
          reachable = set_union(reachable, filter ? set_intersect(jt.clique(s), *filter) : jt.clique(s));
        }

        // Store the reachable set in the appropriate variable
        if (s < t)
          jt.graph[e].forward_reachable = reachable;
        else
          jt.graph[e].reverse_reachable = reachable;
      }
    };

    /**
     * A functor that given a vertex, returns the corresponding clique.
     */
    class clique_functor
      : public std::unary_function<vertex, const node_set&> {
      const junction_tree* jt_ptr;
    public:
      clique_functor(const junction_tree* jt_ptr) : jt_ptr(jt_ptr) { }
      const node_set& operator()(vertex v) const { return jt_ptr->clique(v); }
    };

    /**
     * A functor that given an edge, returns the corresponding separator.
     */
    class separator_functor
      : public std::unary_function<edge, const node_set&> {
      const junction_tree* jt_ptr;
    public:
      separator_functor(const junction_tree* jt_ptr) : jt_ptr(jt_ptr) { }
      const node_set& operator()(edge e) const { return jt_ptr->separator(e); }
    };

    /**
     * A functor that, given a vertex, computes the negative separator size.
     */
    class sepsize_functor : public std::unary_function<edge, int> {
      const junction_tree* jt;
    public:
      sepsize_functor(const junction_tree* jt) : jt(jt) { }
      int operator()(edge e) const {
        vertex u = e.source(), v = e.target();
        return -int(set_intersect(jt->clique(u),jt->clique(v)).size());
      }
    };

  }; // class junction_tree

  /**
   * Prints a human-readable representation of the junction tree to
   * the supplied output stream.
   * \relates junction_tree
   */
  template <typename Node, typename VP, typename EP>
  std::ostream&
  operator<<(std::ostream& out, const junction_tree<Node, VP, EP>& jt) {
    jt.print(out);
    return out;
  }

} // namespace sill


namespace boost {

  //! A traits class that lets junction_tree work in BGL algorithms
  //! (inherits from the traits class for the underlying undirected_graph)
  template <typename Node, typename VP, typename EP>
  struct graph_traits< sill::junction_tree<Node, VP, EP> >
    : public graph_traits<typename sill::junction_tree<Node, VP, EP>::graph_type>
    { };

} // namespace boost

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_JUNCTION_TREE_HPP
