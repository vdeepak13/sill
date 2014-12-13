#ifndef SILL_UNDIRECTED_GRAPH_HPP
#define SILL_UNDIRECTED_GRAPH_HPP
#include <set>
#include <list>
#include <iterator>
#include <iosfwd>
#include <map>

#include <boost/tuple/tuple_comparison.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered_map.hpp>

#include <sill/global.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/graph/undirected_edge.hpp>
#include <sill/graph/boost_graph_helpers.hpp>
#include <sill/iterator/map_key_iterator.hpp>
#include <sill/range/concepts.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>

namespace sill {


  //============================================================================
  /**
   * Undirected Graph Class
   * \ingroup graph_types
   */
  template < typename Vertex,
             typename VertexProperty = void_, 
             typename EdgeProperty = void_>
  class undirected_graph {
  private:

    // Private type and member declarations
    //==========================================================================

    //! This map type is used to associate edge data with each target
    typedef boost::unordered_map<Vertex, EdgeProperty*> edge_property_map;
    //    typedef std::map<Vertex, EdgeProperty*> edge_property_map;
    
    //! A struct with the data associated with each edge
    struct vertex_data {
      VertexProperty property;
      edge_property_map neighbors;
      vertex_data() { }
      vertex_data(const vertex_data& other) : 
        property(other.property), neighbors(other.neighbors) { }
    };

    //! The type of the map that associates all the vertices with their 
    //! vertex_data.  
    typedef boost::unordered_map<Vertex, vertex_data> vertex_data_map;
    //    typedef std::map<Vertex, vertex_data> vertex_data_map;

    // data members ------------------------------------------------------------
   
    /**
     * This is a map from Vertex to Vertex Data which contains the property
     * associated with each vertex and maps to edge data for the edges 
     * emanating from parents and leading to children.
     */
    vertex_data_map data_map;

    //! The total number of edges in the graph
    size_t edge_count;

  public:
    // Public type declerations -----------------------------------------------
    // Binding template arguments
    typedef Vertex vertex;                       //!< The vertex type
    typedef sill::undirected_edge<Vertex> edge;   //!< The edge type
    typedef VertexProperty vertex_property; //!< Data associated with vertices
    typedef EdgeProperty edge_property;     //!< Data associated with edges

    // Forward declerations. See bottom of the class for implemenations
    class edge_iterator;     //!< Iterator over all edges of the graph
    class in_edge_iterator;  //!< Iterator over incoming edges to a node
    class out_edge_iterator; //!< Iterator over outgoing edges from a node

    //! Iterator over all vertices
    typedef map_key_iterator<vertex_data_map>   vertex_iterator;

    //! Iterator over neighbors of a single vertex
    typedef map_key_iterator<edge_property_map> neighbor_iterator;

    // Constructors and destructors
    //==========================================================================
  public:
    /**
     * Create an empty graph.
     * \todo The max load factor of the internal hash map is set to 1?
     */
    undirected_graph() : data_map(), edge_count(0) { 
      // data_map.max_load_factor(1);
    }

    /**
     * Create a graph from a list of pairs of vertices
     */
    template <typename Range>
    undirected_graph(const Range& edges, typename Range::iterator* = 0)
      : data_map(), edge_count(0) {
      typedef std::pair<vertex, vertex> vertex_pair;
      //concept_assert((InputRangeConvertible<Range, vertex_pair>));

      // data_map.max_load_factor(1);
      foreach(vertex_pair vp, edges) {
        if(!contains(vp.first))  add_vertex(vp.first);
        if(!contains(vp.second)) add_vertex(vp.second);
        add_edge(vp.first, vp.second);
      }
    }
    
    //! copy constructor
    undirected_graph(const undirected_graph& g) {
      *this = g;
    }

    //! destructor
    ~undirected_graph() {
      free_edge_data();
    }

    //! Assignment
    undirected_graph& operator=(const undirected_graph& g) {
      if (this == &g) return *this;
      free_edge_data();
      data_map = g.data_map;
      edge_count = g.edge_count;
      foreach(edge e, edges()) {
        if(e.m_property != NULL) {
          vertex u = e.source(), v = e.target();
          edge_property* p_ptr = 
            new edge_property(*static_cast<edge_property*>(e.m_property));
          data_map[u].neighbors[v] = p_ptr;
          data_map[v].neighbors[u] = p_ptr;
        }
      }
      return *this;
    }

    //! Swaps two graphs in constant time.
    void swap(undirected_graph& other) {
      data_map.swap(other.data_map);
      std::swap(edge_count, other.edge_count);
    }

    // Accessors
    //==========================================================================

    //! Returns an ordered set of all vertices
    std::pair<vertex_iterator, vertex_iterator>
    vertices() const {
      return std::make_pair(vertex_iterator(data_map.begin()),
                            vertex_iterator(data_map.end()));
    }
    
    //! Returns the vertices adjacent to u
    std::pair<neighbor_iterator, neighbor_iterator>
    neighbors(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(neighbor_iterator(vdata.neighbors.begin()),
                            neighbor_iterator(vdata.neighbors.end()));
    }

    //! Returns the vertices adjacent to u
    std::pair<neighbor_iterator, neighbor_iterator>
    adjacent_vertices(const vertex& u) const {
      return neighbors(u);
    }      
    
    //! Returns all edges in the graph
    std::pair<edge_iterator, edge_iterator>
    edges() const {
      return std::make_pair(edge_iterator(data_map.begin(), data_map.end()),
                            edge_iterator(data_map.end(), data_map.end()));
    }

    //! Returns all edges connected to u in the form.
    std::pair<out_edge_iterator, out_edge_iterator>
    edges(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(out_edge_iterator(u, vdata.neighbors.begin()),
                            out_edge_iterator(u, vdata.neighbors.end()));
    }


    //! Returns the edges incoming to a vertex
    std::pair<in_edge_iterator, in_edge_iterator>
    in_edges(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(in_edge_iterator(u, vdata.neighbors.begin()),
                            in_edge_iterator(u, vdata.neighbors.end()));
    }

    //! Returns the outgoing edges from a vertex
    std::pair<out_edge_iterator, out_edge_iterator>
    out_edges(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(out_edge_iterator(u, vdata.neighbors.begin()),
                            out_edge_iterator(u, vdata.neighbors.end()));
    }

    //! Returns true iff the graph contains the given vertex
    bool contains(const vertex& u) const {
      return data_map.find(u) != data_map.end();
    }

    //! Returns true iff the graph contains the given vertices
    bool contains(const std::set<vertex>& vertices) const {
      foreach(vertex v, vertices)
        if (!contains(v)) return false;
      return true;
    }
    
    //! Returns true if the graph contains an undirected edge {u, v}
    bool contains(const vertex& u, const vertex& v) const {
      typename vertex_data_map::const_iterator it = data_map.find(u);
      return it != data_map.end() && 
        (it->second.neighbors.find(v) != it->second.neighbors.end());     
    }
    
    //! Returns true if the graph contains an undirected edge
    bool contains(const edge& e) const {
      return contains(e.source(), e.target());
    }

    //! Returns an undirected edge with e.source()==u and e.target()==v.
    //! The edge must exist.
    edge get_edge(const vertex& u,  const vertex& v) const {
      const vertex_data& vdata = find_vertex_data(u);
      typename edge_property_map::const_iterator it = vdata.neighbors.find(v);
      // Verify that the edge esists in the graph
      assert(it != vdata.neighbors.end());
      return edge(u, v, it->second);
    }

    //! Returns the number of edges adjacent to a vertex
    size_t in_degree(const vertex& u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns the number of edges adjacent to a vertex
    size_t out_degree(const vertex& u) const {
      return find_vertex_data(u).neighbors.size();
    }
    
    //! Returns the number of edges adjacent to a vertex
    size_t degree(const vertex& u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns true if the graph has no vertices
    bool empty() const {
      return data_map.size() == 0;
    }

    //! Returns the number of vertices
    size_t num_vertices() const {
      return data_map.size();
    }

    //! Returns the number of edges
    size_t num_edges() const {
      return edge_count;
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u)
    edge reverse(const edge& e) const { 
      return edge(e.target(), e.source(), e.m_property); 
    }

    //! Returns the property associated with a vertex
    const vertex_property& operator[](const vertex& u) const {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a vertex
    vertex_property& operator[](const vertex& u) {
      // Can't use find_vertex_data function because return type can't be
      // constant.
      typename vertex_data_map::iterator iter(data_map.find(u));
      // The vertex must already be present in the graph
      assert(iter != data_map.end());
      return iter->second.property;
    }
    
    //! Returns the property associated with an edge
    const edge_property& operator[](const edge& e) const {
      return *static_cast<edge_property*>(e.m_property);
    }

    //! Returns the property associated with an edge
    edge_property& operator[](const edge& e) {
      return *static_cast<edge_property*>(e.m_property);
    }

    //! Returns a null vertex
    static vertex null_vertex() { return Vertex(); }

    //! Compares the graph strucutre and the vertex & edge properties.
    //! The property types must support operator!=()
    bool operator==(const undirected_graph& other) const {
      if (num_vertices() != other.num_vertices() ||
          num_edges() != other.num_edges()) {
        return false;
      }
      foreach(typename vertex_data_map::const_reference vp, data_map) {
        const vertex_data* data_other = get_ptr(other.data_map, vp.first);
        if (!data_other || vp.second.property != data_other->property) {
          return false;
        }
        foreach(typename edge_property_map::const_reference ep, vp.second.neighbors) {
          EdgeProperty* const* ep_other = get_ptr(data_other->neighbors, ep.first);
          if (!ep_other || *ep.second != **ep_other) {
            return false;
          }
        }
      }
      return true;
    }

    //! Inequality comparison
    bool operator!=(const undirected_graph& other) const {
      return !(*this == other);
    }
    
    // Modifications
    //==========================================================================
    /**
     * Adds a vertex to a graph and associate the property with that vertex.  
     * If no property is provided then the default property is used.
     * \returns true if the vertex was already present 
     */
    bool add_vertex(const vertex& u, 
                    const vertex_property& p = vertex_property()) {
      bool is_present = contains(u);
      data_map[u].property = p;
      return is_present;
    }

    /**
     * Adds an edge to a graph
     * If the edge already exists, overwrites the data.
     */
    std::pair<edge, bool>
    add_edge(const vertex& u, const vertex& v,
             const edge_property& p = edge_property()) {
      bool is_present = contains(u, v);
      if(  is_present ) { 
        delete data_map[u].neighbors[v];
        --edge_count;
      }
      edge_property* p_ptr = new edge_property(p);
      data_map[u].neighbors[v] = p_ptr;
      data_map[v].neighbors[u] = p_ptr;
      // Count the edge
      ++edge_count;
      return std::make_pair(edge(u, v, p_ptr), is_present);
      // FIXME: should return false if already exists
    }

    /**
     * Make a clique from a beginning and an end iterator
     * \todo optimize this function (don't call add_edge)
     */
    template <typename Iterator>
    void make_clique(const Iterator begin, const Iterator end) {
//      concept_assert((InputIterator<Iterator>));
      Iterator cur = begin;
      for (cur = begin; cur != end; ++cur) {
        Iterator next = cur;
        while (++next != end) add_edge(*cur, *next);
      }
    }

    /**
     * Make a clique from a range of vertices. 
     * \todo optimize this function (don't call add_edge)
     */
    template <typename Range>
    void make_clique(const Range& vertex_range) {
      //concept_assert((ReadableForwardRange<Range>));
      make_clique(boost::begin(vertex_range), 
                  boost::end(vertex_range));
    }

    //! Removes a vertex from the graph and all its incident edges
    void remove_vertex(const vertex& u) {
      clear_edges(u); 
      // not sure if data.neighbors.clear() + data_map.erase(u)
      // is less efficient than data_map.erase(u) alone
      data_map.erase(u);
    }

    //! Removes an undirected edge {u, v}
    void remove_edge(const vertex& u, const vertex& v) {
      //      assert(contains(u,v));
      edge_property* p_ptr = data_map[u].neighbors[v];
      if (p_ptr != NULL) delete p_ptr;
      data_map[u].neighbors.erase(v);
      data_map[v].neighbors.erase(u);
      --edge_count;
    }

    //! Removes all edges incident to a vertex
    void clear_edges(const vertex& u) {
      typename vertex_data_map::iterator iter = data_map.find(u);
      // Vertex is not present in graph
      assert(iter != data_map.end());
      vertex_data& data = iter->second;
      // Remove all external connections to u
      typedef std::pair<vertex, edge_property*> edge_tuple;
      edge_count -= data.neighbors.size();
      foreach(edge_tuple e, data.neighbors) {
        if(e.first != u) data_map[e.first].neighbors.erase(u);
        if(e.second != NULL) delete e.second;
      }
      data.neighbors.clear();
    }

    //! Removes all edges from the graph
    void clear_edges() {
      free_edge_data();
      // Clear the edge maps
      foreach(typename vertex_data_map::reference p, data_map) 
        p.second.neighbors.clear();
      edge_count = 0;
    }

    //! Removes all vertices and edges from the graph
    void clear() {
      free_edge_data();
      data_map.clear();
      edge_count = 0;
    }

    //! Saves the graph to an archive
    void save(oarchive& ar) const {
      ar << num_vertices();
      ar << num_edges();
      foreach(vertex v, vertices())
        ar << v << operator[](v);
      foreach(edge e, edges())
        ar << e.source() << e.target() << operator[](e);
    }
    
    //! Loads the graph from an archive
    void load(iarchive& ar) {
      clear();
      size_t num_vertices, num_edges;
      vertex u, v;
      ar >> num_vertices;
      ar >> num_edges;
      while (num_vertices-- > 0) {
        ar >> v; add_vertex(v);
        ar >> operator[](v);
      }
      while (num_edges -- > 0) {
        ar >> u >> v; 
        ar >> operator[](add_edge(u, v).first);
      }
    }

    // Private helper functions and serialization
    //==========================================================================
  private: 
    const vertex_data& find_vertex_data(const vertex& v) const { 
      typename vertex_data_map::const_iterator vdata(data_map.find(v));
      // The vertex does not exist in this graph
      assert(vdata != data_map.end());
      return vdata->second;
    }

    //! Free the memory associated with each edge
    void free_edge_data() {
      foreach(edge e, edges()) {
        if(e.m_property != NULL)
          delete static_cast<edge_property*>(e.m_property);
      }
    }

    // Public member classes
    //==========================================================================
  public:
    class edge_iterator : 
      public std::iterator<std::forward_iterator_tag, edge> {
    public:
      typedef edge reference;
      typedef typename vertex_data_map::const_iterator master_iterator;
      typedef typename edge_property_map::const_iterator slave_iterator;

    private:
      master_iterator master_it, master_end;
      slave_iterator slave_it, slave_end;
      /**
       * If slave_it does not point to an edge (u, v) with u <= v,
       * this function advances the master / slave iterators
       * until we do or we reach the end of iteration. 
       * This operation can take O(|V|) time
       *
       * Note: Under MSVC with iterator debugging turned on,
       * it is an error to compare iterators that have been 
       * default-initialized. Therefore, this function must
       * be called only when slave_it and slave_end are valid.
       */
      void maintain_invariant() {
        while (slave_it != slave_end && master_it->first > slave_it->first)
          ++slave_it;
        while(slave_it == slave_end && master_it != master_end &&
              ++master_it != master_end) {
          slave_it = master_it->second.neighbors.begin();
          slave_end = master_it->second.neighbors.end();
          // Iterate until we find an edge (u, v) with u <= v
          while (slave_it != slave_end && master_it->first > slave_it->first)
            ++slave_it;
        }
      }
    public:
      edge_iterator() {}
      edge_iterator(master_iterator master_it,
                    master_iterator master_end)
      : master_it(master_it), master_end(master_end) {
        if(master_it != master_end) {        
          slave_it = master_it->second.neighbors.begin();
          slave_end = master_it->second.neighbors.end();
          maintain_invariant();
        }
      }
      edge operator*() const {
        return edge(master_it->first, slave_it->first, slave_it->second);
      }
      edge_iterator& operator++() {
        // If we can advance within the same master then do so
        if(slave_it != slave_end) ++slave_it;
        // If we are not pointing to a valid edge move the master and dest
        // until we are or we fail
        maintain_invariant();
        return *this;
      }     
      edge_iterator operator++(int) {
        edge_iterator copy = *this;
        operator++();
        return copy;
      }
      bool operator==(const edge_iterator& o) const {
        return (master_it == master_end && o.master_it == o.master_end) ||
          ( (master_it == o.master_it) &&
            (master_end == o.master_end) &&
            (slave_it == o.slave_it) &&
            (slave_end == o.slave_end) );
      }     
      bool operator!=(const edge_iterator& other) const {
        return !(operator==(other));
      }
    };

    class out_edge_iterator : 
      public std::iterator<std::forward_iterator_tag, edge> {
    public:
      typedef edge reference;
      typedef typename edge_property_map::const_iterator iterator;
    private:
      vertex source;
      iterator it;
    public:
      out_edge_iterator() {}
      out_edge_iterator(const vertex& source, const iterator& it)
        : source(source), it(it) {}
      edge operator*() { return edge(source, it->first, it->second); }
      out_edge_iterator& operator++() { ++it; return *this; }
      out_edge_iterator operator++(int) {
        out_edge_iterator copy(*this);
        operator++();
        return copy;
      }
      bool operator==(const out_edge_iterator& o) const {
        return it == o.it;
      }     
      bool operator!=(const out_edge_iterator& o) const {
        return !(operator==(o));
      }
    };

    class in_edge_iterator : 
      public std::iterator<std::forward_iterator_tag, edge> {
    public:
      typedef edge reference;
      typedef typename edge_property_map::const_iterator iterator;
    private:
      vertex source;
      iterator it;
    public:
      in_edge_iterator() {}
      in_edge_iterator(const vertex& source, const iterator& it)
        : source(source), it(it) {}
      edge operator*() const { return edge(it->first, source, it->second); }
      in_edge_iterator& operator++() { ++it; return *this; }
      in_edge_iterator operator++(int) {
        in_edge_iterator copy(*this);
        operator++();
        return copy;
      }
      bool operator==(const in_edge_iterator& o) const {
        return it == o.it;
      }     
      bool operator!=(const in_edge_iterator& o) const {
        return !(operator==(o));
      }
    };

  }; // class undirected_graph
  
  //! Prints the graph to an output stream
  //! \relates undirected_graph
  template <typename Vertex, typename VP, typename EP>
  std::ostream& operator<<(std::ostream& out,
                           const undirected_graph<Vertex, VP, EP>& g) {
    out << "Vertices" << std::endl;
    foreach(Vertex v, g.vertices())
      out << v << ": " << g[v] << std::endl;
    out << "Edges" << std::endl;
    foreach(undirected_edge<Vertex> e, g.edges()) 
      out << e << std::endl;
    return out;
  }

} // namespace sill


namespace boost {

  //! Type declarations that let our graph structure work in BGL algorithms
  template <typename Vertex, typename VP, typename EP>
  struct graph_traits< sill::undirected_graph<Vertex, VP, EP> > {
    
    typedef sill::undirected_graph<Vertex, VP, EP> graph_type;

    typedef typename graph_type::vertex             vertex_descriptor;
    typedef typename graph_type::edge               edge_descriptor;
    typedef typename graph_type::vertex_iterator    vertex_iterator;
    typedef typename graph_type::neighbor_iterator  adjacency_iterator;
    typedef typename graph_type::edge_iterator      edge_iterator;
    typedef typename graph_type::out_edge_iterator  out_edge_iterator;
    typedef typename graph_type::in_edge_iterator   in_edge_iterator;

    typedef undirected_tag                          directed_category;
    typedef disallow_parallel_edge_tag              edge_parallel_category;

    struct traversal_category :
      public virtual boost::vertex_list_graph_tag,
      public virtual boost::incidence_graph_tag,
      public virtual boost::adjacency_graph_tag,
      public virtual boost::edge_list_graph_tag { };

    typedef size_t vertices_size_type;
    typedef size_t edges_size_type;
    typedef size_t degree_size_type;

    static vertex_descriptor null_vertex() { return vertex_descriptor(); }

  };

} // namespace boost

#include <sill/macros_undef.hpp>

#endif
