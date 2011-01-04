#ifndef SILL_DIRECTED_GRAPH_HPP
#define SILL_DIRECTED_GRAPH_HPP

#include <map>
#include <list>
#include <iterator>
#include <iosfwd>

#include <boost/graph/graph_traits.hpp>
#include <boost/unordered_map.hpp>

#include <sill/global.hpp>
#include <sill/graph/directed_edge.hpp>
#include <sill/graph/boost_graph_helpers.hpp>
#include <sill/iterator/map_key_iterator.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  // ===========================================================================
  /**
   * Directed Graph Class
   * \ingroup graph_types
   * \see Graph
   */
  template < typename Vertex,
             typename VertexProperty = void_, 
             typename EdgeProperty = void_>
  class directed_graph {

    // Private type declarations and data members
    //==========================================================================
  private:
    /** 
     * This map type is used to associate edge data with each target. 
     * The "edgeness" of this property map only makes sense when considering
     * the vertex_data map.
     */
    typedef boost::unordered_map<Vertex, EdgeProperty*> edge_property_map;

    /**
     * A struct with the data associated with each vertex.  This structure
     * store the property associated with the vertex as well as edge property
     * maps which map parent and child vertices to the edge property associated
     * with the edge to the parent or child.  The parent and child language is 
     * used to mean in and out edges respectively and has nothing to do with
     * trees.
     */
    struct vertex_data {
      VertexProperty property;
      edge_property_map parents;
      edge_property_map children;
      vertex_data() { }
      vertex_data(const vertex_data& other) :
        property(other.property), parents(other.parents), 
        children(other.children) { }
    };

    /**
     * The type of the map that associates all the vertices with their 
     * vertex_data. This represents the major datatype for storing graphs
     */
    typedef boost::unordered_map<Vertex, vertex_data> vertex_data_map;

    // data members ------------------------------------------------------------
    /**
     * This is a map from Vertex to Vertex Data which contains the property
     * associated with each vertex and maps to edge data for the edges 
     * emanating from parents and leading to children.
     */
    vertex_data_map data_map;
    
    //! The total number of directed edges in the graph
    size_t edge_count;

    // Public type declerations
    //==========================================================================
  public:
    
    // Binding template arguments
    typedef Vertex vertex;                  //!< The vertex type
    typedef sill::directed_edge<Vertex> edge;     //!< The edge type
    typedef VertexProperty vertex_property; //!< Data associated with vertices
    typedef EdgeProperty edge_property;     //!< Data associated with Edges
    
    // Forward declerations. See bottom of the class for implemenations
    class edge_iterator;     //!< Iterator over all edges of the graph
    class in_edge_iterator;  //!< Iterator over incoming edges to a node
    class out_edge_iterator; //!< Iterator over outgoing edges from a node
    
    //! Iterator over all vertices
    typedef map_key_iterator<vertex_data_map>   vertex_iterator;

    //! Iterator over the neighbors of a single vertex
    typedef map_key_iterator<edge_property_map> neighbor_iterator;


    // Constructors and destructors
    //==========================================================================
  public:
    //! Create an empty graph.
    directed_graph() : data_map(), edge_count(0) { }

    //! Create a graph from a list of pairs.  
    template <typename Range>
    directed_graph(const Range& edges, typename Range::iterator* = 0)
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
    
    //! Copy constructor
    directed_graph(const directed_graph& g) 
      : data_map(g.data_map), edge_count(g.edge_count) {
      //! Copy the edge properties 
      foreach(edge e, edges()) {
        if(e.m_property != NULL) {
          vertex source = e.source(), target = e.target();
          edge_property* p_ptr = 
            new edge_property(*static_cast<edge_property*>(e.m_property));
          data_map[source].children[target] = p_ptr;
          data_map[target].parents[source]  = p_ptr;
        }
      }
    }

    //! Destructor
    ~directed_graph() {
      // Free the memory associated with each edge
      foreach(edge e, edges()) {
        if(e.m_property != NULL) {
          delete static_cast<edge_property*>(e.m_property);
        }
      }
    }

    //! Assignment operator
    directed_graph& operator=(const directed_graph& other) {
      // Destroy the information associated with this graph
      clear();
      // Copy all the info in other
      data_map = other.data_map;
      edge_count = other.edge_count;
      // Copy the edge properties 
      foreach(edge e, edges()) {
        if(e.m_property != NULL) {
          vertex source = e.source(), target = e.target();
          edge_property* p_ptr = 
            new edge_property(*static_cast<edge_property*>(e.m_property));
          data_map[source].children[target] = p_ptr;
          data_map[target].parents[source]  = p_ptr;
        }
      }
      return *this;
    }

    //! Swaps two graphs in constant time.
    void swap(directed_graph& other) {
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
    
    //! Returns the parents of u
    std::pair<neighbor_iterator, neighbor_iterator>
    parents(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(neighbor_iterator(vdata.parents.begin()),
                            neighbor_iterator(vdata.parents.end()));
    }

    //! Returns the children of u
    std::pair<neighbor_iterator, neighbor_iterator>
    children(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(neighbor_iterator(vdata.children.begin()),
                            neighbor_iterator(vdata.children.end()));
    }

    //! Returns all edges in the graph
    std::pair<edge_iterator, edge_iterator>
    edges() const {
      return std::make_pair(edge_iterator(data_map.begin(), data_map.end()),
                            edge_iterator(data_map.end(), data_map.end()));
    }

    //! Returns the edges incoming to a vertex
    std::pair<in_edge_iterator, in_edge_iterator>
    in_edges(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(in_edge_iterator(u, vdata.parents.begin()),
                            in_edge_iterator(u, vdata.parents.end()));
    }

    //! Returns the outgoing edges from a vertex
    std::pair<out_edge_iterator, out_edge_iterator>
    out_edges(const vertex& u) const {
      const vertex_data& vdata(find_vertex_data(u));
      return std::make_pair(out_edge_iterator(u, vdata.children.begin()),
                            out_edge_iterator(u, vdata.children.end()));
    }

    //! Returns true iff the graph contains the given vertex
    bool contains(const vertex& u) const {
      return data_map.find(u) != data_map.end();
    }
    
    //! Returns true if the graph contains a directed edge (u, v)
    bool contains(const vertex& source, const vertex& target) const {
      typename vertex_data_map::const_iterator it = data_map.find(source);
      return it != data_map.end() && 
        (it->second.children.find(target) != it->second.children.end());     
    }
    
    //! Returns true if the graph contains a directed edge
    bool contains(const edge& e) const {
      return contains(e.source(), e.target());
    }

    //! Returns an edge between two vertices. The edge must exist.
    edge get_edge(const vertex& source, const vertex& target) const {
      const vertex_data& vdata = find_vertex_data(source);
      typename edge_property_map::const_iterator it = 
        vdata.children.find(target);
      // Verify that edge exists in graph
      assert(it != vdata.children.end());
      return edge(source, target, it->second);
    }

    /**
     * Returns the number of incoming edges to a vertex.  The vertex must
     * already be present in the graph,
     */
    size_t in_degree(const vertex& u) const {
      return find_vertex_data(u).parents.size();
    }

    /**
     * Returns the number of outgoing edges to a vertex.  The vertex must
     * already be present in the graph.
     */
    size_t out_degree(const vertex& u) const {
      return find_vertex_data(u).children.size();
    }
    
    /**
     * Returns the total number of edges adjacent to a vertex.  The vertex must
     * already be present in the graph.
     */
    size_t degree(const vertex& u) const {
      const vertex_data& vdata = find_vertex_data(u);
      return vdata.parents.size() + vdata.children.size();
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

    //! Given a directed edge (u, v), returns a directed edge (v, u)
    //! The edge (v, u) must exist.
    edge reverse(const edge& e) const { 
      return get_edge(e.target(), e.source()); 
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

    //! Returns the property associated with an edge.
    //! The edge (u, v) must exist.
    const edge_property& operator()(const vertex& u, const vertex& v) const {
      return *static_cast<edge_property*>(get_edge(u, v).m_property);
    }

    //! Returns the property associated with an edge.
    //! The edge (u, v) is added if necessary.
    edge_property& operator()(const vertex& u, const vertex& v) {
      if (contains(u, v))
        return *static_cast<edge_property*>(get_edge(u, v).m_property);
      else
        return *static_cast<edge_property*>(add_edge(u, v).first.m_property);
    }

    //! Returns a null vertex.
    static vertex null_vertex() { return Vertex(); }
    
    // Modifications
    //==========================================================================
    /**
     * Adds a vertex to a graph.
     * \todo Maybe we should enforce here that u does not exist yet?
     *       or return true/false based on whether u exists?
     */
    bool add_vertex(const vertex& u, 
                    const vertex_property& p = vertex_property()) {
      bool is_present = contains(u);
      data_map[u].property = p;
      return is_present;
    }

    /**
     * Adds an edge to a graph.
     *
     * \todo Do we require here that source and target already exist? 
     *       If not, we have to make sure that the corresponding vertex_property
     *       gets default-initialized (relevant for primitive datatypes).
     * \todo Is this an old comment?  I don't think it is a problem any more:
     *       This will memory leak if (source,target) already exists.
     * \todo Make this more efficient by avoiding duplicate hash lookups.
     */
    std::pair<edge, bool>
    add_edge(const vertex& source, const vertex& target,
             const edge_property& p = edge_property()) {
      bool already_existed = contains(source, target);
      if( already_existed ) {
        delete data_map[source].children[target];
        --edge_count;
      }
      // Allocate a new block of memory for the edge_property p
      edge_property* p_ptr = new edge_property(p); 
      // Add the property as a child of source
      data_map[source].children[target] = p_ptr;
      data_map[target].parents[source]  = p_ptr;
      // Count the edge
      ++edge_count;
      return std::make_pair(edge(source, target, p_ptr), already_existed);
    }

    /**
     * Make a clique from a beginning and an end iterator
     * \todo optimize this function (don't call add_edge)
     */
    template <typename Iterator>
    void make_clique(const Iterator begin, const Iterator end) {
      //concept_assert((InputIterator<Iterator>));
      Iterator cur = begin;
      for (cur = begin; cur != end; ++cur) {
        Iterator next = cur;
        while (++next != end) {
          add_edge(*cur, *next);
          add_edge(*next, *cur);
        }
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
      // Remove all external connections to u
      clear_edges(u);      
      // Remove the vertex data
      data_map.erase(u);
    }

    //! Removes a directed edge (u, v)
    void remove_edge(const vertex& source, const vertex& target) {
      // Requie that the edge be present before removal
      assert(contains(source,target));
      // Delete teh edge property
      delete data_map[source].children[target];
      // Remove from maps
      data_map[source].children.erase(target);
      data_map[target].parents.erase(source);
      --edge_count;
    }

    //! Removes all edges incident to a vertex
    void clear_edges(const vertex& u) {
      clear_in_edges(u);
      clear_out_edges(u);
    }

    //! Removes all edges incoming to a vertex
    void clear_in_edges(const vertex& u) {
      typename vertex_data_map::iterator iter = data_map.find(u);
      // Vertex is not present in graph
      assert(iter != data_map.end());
      vertex_data& data = iter->second;
      // Remove all edges to parents of u
      typedef std::pair<vertex, edge_property*> edge_tuple;
      edge_count -= data.parents.size();
      // Disconnect from parents
      foreach(edge_tuple e, data.parents) {
        data_map[e.first].children.erase(u);
        if(e.second != NULL) delete e.second;
      }
      data.parents.clear();
    }

    //! Removes all edges outgoing from a vertex
    void clear_out_edges(const vertex& u) {
      typename vertex_data_map::iterator iter = data_map.find(u);
      // Vertex is not present in graph
      assert(iter != data_map.end());
      vertex_data& data = iter->second;
      // Remove all external connections to u
      typedef std::pair<vertex, edge_property*> edge_tuple;
      edge_count -= data.children.size();
      // Disconnect from children
      foreach(edge_tuple e, data.children) {
        data_map[e.first].parents.erase(u);
        if(e.second != NULL) delete e.second;
      }
      data.children.clear();
    }

    //! Removes all edges from the graph
    void clear_edges() {
      free_edge_data();
      // Clear the edge maps
      foreach(typename vertex_data_map::reference p, data_map) {
        p.second.parents.clear();
        p.second.children.clear();
      }
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

    // Private functions and serialization
    //==========================================================================
  private: 
    const vertex_data& find_vertex_data(const vertex& v) const { 
      typename vertex_data_map::const_iterator vdata(data_map.find(v));
      // Verifyt hat the vertex data is found
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
       * If slave_it does not point to a valid edge, this function 
       * advances the master / slave iterators until we do 
       * or we reach the end of iteration. 
       * This operation can take O(|V|) time
       *
       * Note: Under MSVC with iterator debugging turned on,
       * it is an error to compare iterators that have been 
       * default-initialized. Therefore, this function must
       * be called only when slave_it and slave_end are valid.
       */
      void maintain_invariant() {
        while(slave_it == slave_end && master_it != master_end &&
              ++master_it != master_end) {
          slave_it = master_it->second.children.begin();
          slave_end = master_it->second.children.end();
        }  
      }
    public:
      edge_iterator(master_iterator master_it,
                    master_iterator master_end)
      : master_it(master_it), master_end(master_end) {
        if(master_it != master_end) {        
          slave_it = master_it->second.children.begin();
          slave_end = master_it->second.children.end();
          maintain_invariant();
        }
      }
      edge operator*() {
        return edge(master_it->first, slave_it->first, slave_it->second);
      }
      edge_iterator& operator++() {
        // If we can advance within the same master then do so
        assert(slave_it != slave_end);
        ++slave_it;
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
      in_edge_iterator(const vertex& source, const iterator& it)
        : source(source), it(it) {}
      edge operator*() { return edge(it->first, source, it->second); }
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


  }; // class directed_graph

  //! Print graphs to an output tream (for debugging purposes).
  //! \relates directed_graph
  template <typename Vertex, typename VP, typename EP>
  std::ostream& operator<<(std::ostream& out,
                           const directed_graph<Vertex, VP, EP>& g) {
    out << "Vertices" << std::endl;
    foreach(Vertex v, g.vertices())
      out << v << ": " << g[v] << std::endl;
    out << "Edges" << std::endl;
    foreach(directed_edge<Vertex> e, g.edges()) 
      out << e << std::endl;
    return out;
  } 

} // namespace sill


namespace boost {

  //! Type declarations that let our graph structure work in BGL algorithms
  template <typename Vertex, typename VP, typename EP>
  struct graph_traits< sill::directed_graph<Vertex, VP, EP> > {
    
    typedef sill::directed_graph<Vertex, VP, EP> graph_type;

    typedef typename graph_type::vertex             vertex_descriptor;
    typedef typename graph_type::edge               edge_descriptor;
    typedef typename graph_type::vertex_iterator    vertex_iterator;
    typedef typename graph_type::neighbor_iterator  adjacency_iterator;
    typedef typename graph_type::edge_iterator      edge_iterator;
    typedef typename graph_type::out_edge_iterator  out_edge_iterator;
    typedef typename graph_type::in_edge_iterator   in_edge_iterator;

    typedef directed_tag                            directed_category;
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
