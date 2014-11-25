#ifndef SILL_BIPARTITE_GRAPH_HPP
#define SILL_BIPARTITE_GRAPH_HPP

#include <sill/global.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/iterator/join_iterator.hpp>
#include <sill/iterator/map_key_iterator.hpp>
#include <sill/serialization/serialize.hpp>

#include <boost/unordered_map.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents an undirected bipartite graph. The graph contains
   * two types of vertices (type 1 and type 2), which correspond to the two
   * sides of the partition. These vertices are represented by the template
   * arguments Vertex1 and Vertex2, respectively. These two types _must_ be
   * distinct and not convertible to each other. This requirement is necessary
   * to allow for overload resolution to work in functions, such as neighbors().
   * The class defines the member type vertex, which is effectively a union
   * of the two vertex types.
   *
   * \tparam Vertex1 the type that represents a type-1 vertex
   * \tparam Vertex2 the type that represents a type-2 vertex
   * \tparam VertexProperty the type of data associated with the vertices
   * \tparam EdgeProperty the type of data associated with the edge
   */
  template <typename Vertex1,
            typename Vertex2,
            typename VertexProperty = void_,
            typename EdgeProperty = void_>
  class bipartite_graph {
  private:
    // Private types
    //==========================================================================
    typedef boost::unordered_map<Vertex1, EdgeProperty*> neighbor1_map;
    typedef boost::unordered_map<Vertex2, EdgeProperty*> neighbor2_map;

    struct vertex1_data {
      VertexProperty property;
      neighbor2_map neighbors;
    };

    struct vertex2_data {
      VertexProperty property;
      neighbor1_map neighbors;
    };

    typedef boost::unordered_map<Vertex1, vertex1_data> vertex1_data_map;
    typedef boost::unordered_map<Vertex2, vertex2_data> vertex2_data_map;

    // Public types
    //==========================================================================
  public:
    class vertex {
      Vertex1 v1_;
      Vertex2 v2_;
    public:
      vertex() : v1_(), v2_() { }
      vertex(Vertex1 v1) : v1_(v1), v2_() { }
      vertex(Vertex2 v2) : v1_(), v2_(v2) { }
      Vertex1 v1() const { return v1_; }
      Vertex2 v2() const { return v2_; }
      bool type1() const { return v1_ != Vertex1(); }
      bool type2() const { return v2_ != Vertex2(); }
      bool null() const { return v1_ == Vertex1() && v2_ == Vertex2(); }

      friend bool operator==(const vertex& a, const vertex& b) {
        return a.v1_ == b.v1_ && a.v2_ == b.v2_;
      }
      friend bool operator!=(const vertex& a, const vertex& b) {
        return a.v1_ != b.v1_ || a.v2_ != b.v2_;
      }
      friend std::ostream& operator<<(std::ostream& out, const vertex& u) {
        if (u.type1()) {
          out << "1_" << u.v1();
        } else if (u.type2()) {
          out << "2_" << u.v2();
        } else {
          out << "null";
        }
        return out;
      }
    }; // class vertex

    class edge {
      Vertex1 v1_;
      Vertex2 v2_;
      bool forward_; // true if the edge is from type1 to type2
      const EdgeProperty* property_;
      friend class bipartite_graph;
    public:
      edge()
        : v1_(), v2_(), forward_(true), property_() { }
      edge(Vertex1 v1, Vertex2 v2, const EdgeProperty* property)
        : v1_(v1), v2_(v2), forward_(true), property_(property) { }
      edge(Vertex2 v2, Vertex1 v1, const EdgeProperty* property)
        : v1_(v1), v2_(v2), forward_(false), property_(property) { }
      Vertex1 v1() const {
        return v1_;
      }
      Vertex2 v2() const {
        return v2_;
      }
      std::pair<Vertex1, Vertex2> endpoints() const {
        return std::make_pair(v1_, v2_);
      }      
      vertex source() const {
        return forward_ ? vertex(v1_) : vertex(v2_);
      }
      vertex target() const {
        return forward_ ? vertex(v2_) : vertex(v1_);
      }
      bool is_forward() const {
        return forward_;
      }
      bool operator==(const edge& o) const {
        return v1_ == o.v1_ && v2_ == o.v2_;
      }
      bool operator!=(const edge& o) const {
        return v1_ != o.v1_ || v2_ != o.v2_;
      }
      edge reverse() const {
        edge e = *this;
        e.forward_ = !e.forward_;
        return e;
      }
      friend std::ostream& operator<<(std::ostream& out, const edge& e) {
        out << e.source() << " --  " << e.target();
        return out;
      }
      friend size_t hash_value(const edge& e) {
        return boost::hash_value(std::make_pair(e.v1_, e.v2_));
      }
    }; // class edge

    // properties
    typedef VertexProperty vertex_property;
    typedef EdgeProperty edge_property;
    
    // vertex iterators
    typedef map_key_iterator<vertex1_data_map> vertex1_iterator;
    typedef map_key_iterator<vertex2_data_map> vertex2_iterator;
    typedef map_key_iterator<neighbor1_map> neighbor1_iterator;
    typedef map_key_iterator<neighbor2_map> neighbor2_iterator;
    /*
    typedef join_iterator<vertex1_iterator,vertex2_iterator> vertex_iterator;
    typedef join_iterator<neighbor1_iterator,neighbor2_iterator> neighbor_iterator;
    */
    // todo: customize the return type of the join_iterators
    
    // edge iterators (forward declarations and typedefs)
    class edge_iterator;
    template <typename Target, typename Neighbors> class in_edge_iterator;
    template <typename Source, typename Neighbors> class out_edge_iterator;
    typedef in_edge_iterator<Vertex1, neighbor2_map> in1_edge_iterator;
    typedef in_edge_iterator<Vertex2, neighbor1_map> in2_edge_iterator;
    typedef out_edge_iterator<Vertex1, neighbor2_map> out1_edge_iterator;
    typedef out_edge_iterator<Vertex2, neighbor1_map> out2_edge_iterator;
    
    // Constructors, destructors, and related functions
    //==========================================================================
  public:
    //! Creates an empty graph.
    bipartite_graph()
      : edge_count_(0) { }

    //! Creates a graph from a range of vertex pairs
    template <typename Range>
    bipartite_graph(const Range& edges, typename Range::iterator* = 0)
      : edge_count_(0) {
      typedef std::pair<Vertex1, Vertex2> vertex_pair;
      foreach(vertex_pair p, edges) {
        add_edge(p.first, p.second);
      }
    }

    //! Copy constructor
    bipartite_graph(const bipartite_graph& g) {
      *this = g;
    }

    //! Destructor
    ~bipartite_graph() {
      free_edge_data();
    }

    //! Assignment
    bipartite_graph& operator=(const bipartite_graph& g) {
      if (this == &g) { return *this; }
      free_edge_data();
      data1_ = g.data1_;
      data2_ = g.data2_;
      edge_count_ = g.edge_count_;
      foreach(edge e, edges()) {
        edge_property* ptr = new edge_property(*e.property_);
        data1_[e.v1()].neighbors[e.v2()] = ptr;
        data2_[e.v2()].neighbors[e.v1()] = ptr;
      }
      return *this;
    }

    //! Swap with another graph in constant time
    void swap(bipartite_graph& other) {
      data1_.swap(other.data1_);
      data2_.swap(other.data2_);
      std::swap(edge_count_, other.edge_count_);
    }

    // Accessors
    //==========================================================================
  public:
    //! Returns the range of all type-1 vertices
    std::pair<vertex1_iterator, vertex1_iterator>
    vertices1() const {
      return std::make_pair(vertex1_iterator(data1_.begin()),
                            vertex1_iterator(data1_.end()));
    }

    //! Returns the range of all type-2 vertices
    std::pair<vertex2_iterator, vertex2_iterator>
    vertices2() const {
      return std::make_pair(vertex2_iterator(data2_.begin()),
                            vertex2_iterator(data2_.end()));
    }

    /*
    //! Returns the range of all vertices
    std::pair<vertex_iterator, vertex_iterator>
    vertices() const {
      std::pair<vertex1_iterator, vertex1_iterator> v1 = vertices1();
      std::pair<vertex2_iterator, vertex2_iterator> v2 = vertices2();
      return std::make_pair(vertex_iterator(v1.first, v1.second, v2.first),
                            vertex_iterator(v1.second, v1.second, v2.second));
    }
    */

    //! Returns the type-2 vertices adjacent to a type-1 vertex
    std::pair<neighbor2_iterator, neighbor2_iterator>
    neighbors(Vertex1 u) const {
      const vertex1_data& data = find_vertex_data(u);
      return std::make_pair(neighbor2_iterator(data.neighbors.begin()),
                            neighbor2_iterator(data.neighbors.end()));
    }

    //! Returns the type-1 vertices adjacent to a type-2 vertex
    std::pair<neighbor1_iterator, neighbor1_iterator>
    neighbors(Vertex2 u) const {
      const vertex2_data& data = find_vertex_data(u);
      return std::make_pair(neighbor1_iterator(data.neighbors.begin()),
                            neighbor1_iterator(data.neighbors.end()));
    }

    /*
    //! Returns the vertices adjacent to a vertex
    std::pair<neighbor_iterator, neighbor_iterator>
    neighbors(const vertex& u) const {
      std::pair<neighbor1_iterator, neighbor1_iterator> n1;
      std::pair<neighbor2_iterator, neighbor2_iterator> n2;
      if (u.type1()) { n2 = neighbors(u.v1()); }
      if (u.type2()) { n1 = neighbors(u.v2()); }
      return std::make_pair(vertex_iterator(n1.first, n1.second, n2.first),
                            vertex_iterator(n1.second, n1.second, n2.second));
    }

    //! Returns the vertices adjacent to a vertex
    std::pair<neighbor_iterator, neighbor_iterator>
    neighbors(const vertex& u) const {
      return neighbors(u);
    }
    */

    //! Returns all edges in the graph
    std::pair<edge_iterator, edge_iterator>
    edges() const {
      return std::make_pair(edge_iterator(data1_.begin(), data1_.end()),
                            edge_iterator(data1_.end(), data1_.end()));
    }

    //! Returns the edges incoming to a type-1 vertex
    std::pair<in1_edge_iterator, in1_edge_iterator>
    in_edges(Vertex1 u) const {
      const vertex1_data& data = find_vertex_data(u);
      return std::make_pair(in1_edge_iterator(u, data.neighbors.begin()),
                            in1_edge_iterator(u, data.neighbors.end()));
    }

    //! Returns the edges incoming to a type-2 vertex
    std::pair<in2_edge_iterator, in2_edge_iterator>
    in_edges(Vertex2 u) const {
      const vertex2_data& data = find_vertex_data(u);
      return std::make_pair(in2_edge_iterator(u, data.neighbors.begin()),
                            in2_edge_iterator(u, data.neighbors.end()));
    }
    
    //! Returns the edges outgoing from a type-1 vertex
    std::pair<out1_edge_iterator, out1_edge_iterator>
    out_edges(Vertex1 u) const {
      const vertex1_data& data = find_vertex_data(u);
      return std::make_pair(out1_edge_iterator(u, data.neighbors.begin()),
                            out1_edge_iterator(u, data.neighbors.end()));
    }

    //! Returns the edges outgoing from a type-2 vertex
    std::pair<out2_edge_iterator, out2_edge_iterator>
    out_edges(Vertex2 u) const {
      const vertex2_data& data = find_vertex_data(u);
      return std::make_pair(out2_edge_iterator(u, data.neighbors.begin()),
                            out2_edge_iterator(u, data.neighbors.end()));
    }

    //! Returns true if the graph contains the given type-1 vertex
    bool contains(Vertex1 u) const {
      return data1_.find(u) != data1_.end();
    }

    //! Returns true if the graph contains the given type-2 vertex
    bool contains(Vertex2 u) const {
      return data2_.find(u) != data2_.end();
    }

    //! Returns true if the graph contains the given vertex
    bool contains(const vertex& u) const {
      if (u.type1()) {
        return contains(u.v1());
      } else if (u.type2()) {
        return contains(u.v2());
      } else {
        return false; // the graph shall never contain a null vertex
      }
    }

    //! Returns true if the graph contains an undirected edge {u, v}
    bool contains(Vertex1 u, Vertex2 v) const {
      typename vertex1_data_map::const_iterator it = data1_.find(u);
      return it != data1_.end() &&
        it->second.neighbors.find(v) != it->second.neighbors.end();
    }

    //! Returns true if the graph contains an undirected edge {u, v}
    bool contains(const vertex& u, const vertex& v) const {
      if (u.type1() && v.type2()) {
        return contains(u.v1(), v.v2());
      } else if (v.type1() && u.type2()) {
        return contains(v.v1(), u.v2());
      } else {
        return false; // the graph never links vertices in the same partition
      }
    }

    //! Returns true if the graph contains an undirected edge
    bool contains(const edge& e) const {
      return contains(e.v1(), e.v2());
    }

    //! Returns an undirected edge between u and v. The edge must exist.
    edge get_edge(Vertex1 u, Vertex2 v) const {
      const vertex1_data& data = find_vertex_data(u);
      typename neighbor2_map::const_iterator it = data.neighbors.find(v);
      assert(it != data.neighbors.end());
      return edge(u, v, it->second);
    }

    //! Returns an undirected edge between u and v. The edge must exist.
    edge get_edge(Vertex2 u, Vertex1 v) const {
      const vertex2_data& data = find_vertex_data(u);
      typename neighbor1_map::const_iterator it = data.neighbors.find(v);
      assert(it != data.neighbors.end());
      return edge(u, v, it->second);
    }

    //! Returns the number of edges connected to a type-1 vertex
    size_t degree(Vertex1 u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns the number of edges connected to a type-2 vertex
    size_t degree(Vertex2 u) const {
      return find_vertex_data(u).neighbors.size();
    }

    //! Returns the number of edges connected to a vertex
    size_t degree(const vertex& u) const {
      return u.type1() ? degree(u.v1()) : degree(u.v2());
    }

    //! Returns true if the graph has no vertices
    bool empty() const {
      return data1_.empty() && data2_.empty();
    }

    //! Returns the number of type-1 vertices
    size_t num_vertices1() const {
      return data1_.size();
    }

    //! Returns the number of type-2 vertices
    size_t num_vertices2() const {
      return data2_.size();
    }

    //! Returns the number of vertices
    size_t num_vertices() const {
      return data1_.size() + data2_.size();
    }

    //! Returns the number of edges
    size_t num_edges() const {
      return edge_count_;
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u)
    edge reverse(const edge& e) const {
      return e.reverse();
    }

    //! Returns the property associated with a type-1 vertex
    const vertex_property& operator[](Vertex1 u) const {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a type-2 vertex
    const vertex_property& operator[](Vertex2 u) const {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a vertex
    const vertex_property& operator[](const vertex& u) const {
      return u.type1() ? operator[](u.v1()) : operator[](u.v2());
    }

    //! Returns the property associated with a type-1 vertex
    vertex_property& operator[](Vertex1 u) {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a type-2 vertex
    vertex_property& operator[](Vertex2 u) {
      return find_vertex_data(u).property;
    }

    //! Returns the property associated with a vertex
    vertex_property& operator[](const vertex& u) {
      return u.type1() ? operator[](u.v1()) : operator[](u.v2());
    }

    //! Returns the property associated with an edge
    const edge_property& operator[](const edge& e) const {
      return *e.property_;
    }

    //! Returns the property associated with an edge
    edge_property& operator[](const edge& e) {
      return const_cast<edge_property&>(*e.property_);
    }

    //! Returns the null vertex
    static vertex null_vertex() {
      return vertex();
    }

    //! Compares the graph structure and the vertex & edge properties
    bool operator==(const bipartite_graph& g) const {
      if (num_vertices1() != g.num_vertices1() ||
          num_vertices2() != g.num_vertices2() || 
          num_edges() != g.num_edges()) {
        return false;
      }
      foreach(typename vertex1_data_map::const_reference vp, data1_) {
        const vertex1_data* vp_other = get_ptr(g.data1_, vp.first);
        if (!vp_other || vp.second.property != vp_other->property) {
          return false;
        }
        foreach(typename neighbor2_map::const_reference ep, vp.second.neighbors) {
          EdgeProperty* const* ep_other = get_ptr(vp_other->neighbors, ep.first);
          if (!ep_other || *ep.second != **ep_other) {
            return false;
          }
        }
      }
      foreach(typename vertex2_data_map::const_reference vp, data2_) {
        const vertex2_data* vp_other = get_ptr(g.data2_, vp.first);
        if (!vp_other || vp.second.property != vp_other->property) {
          return false;
        }
      }
      return true;
    }

    //! Compares the graph structure and the vertex & edge properties
    bool operator!=(const bipartite_graph& g) const {
      return !operator==(g);
    }

    //! Prints the graph to an output stream.
    friend std::ostream& operator<<(std::ostream& out, const bipartite_graph& g) {
      out << "Type-1 vertices" << std::endl;
      foreach(Vertex1 u, g.vertices1()) {
        out << u << ": " << g[u] << std::endl;
      }
      out << "Type-2 vertices" << std::endl;
      foreach(Vertex2 u, g.vertices2()) {
        out << u << ": " << g[u] << std::endl;
      }
      out << "Edges" << std::endl;
      foreach(edge e, g.edges()) {
        out << e << std::endl;
      }
      return out;
    }

    //! Prints the degree distribution for the given vertex range
    template <typename VertexIt>
    void print_degree_distribution(std::ostream& out,
                                   std::pair<VertexIt,VertexIt> range) const {
      std::map<size_t, size_t> count;
      while (range.first != range.second) {
        ++count[degree(*range.first)];
        ++range.first;
      }

      typedef std::pair<size_t, size_t> size_pair;
      foreach (size_pair p, count) {
        std::cout << p.first << ' ' << p.second << std::endl;
      }
    }

    // Modifications
    //==========================================================================
    /**
     * Adds a type-1 vertex to a graph and associates a property with that
     * that vertex. If the vertex is already present, its property is 
     * overwritten.
     * \returns true if the vertex was already present
     */
    bool add_vertex(Vertex1 u, const VertexProperty& p = VertexProperty()) {
      assert(u != Vertex1());
      bool is_present = contains(u);
      data1_[u].property = p;
      return is_present;
    }

    /**
     * Adds a type-2 vertex to a graph and associates a property with that
     * that vertex. If the vertex is already present, its property is 
     * overwritten.
     * \returns true if the vertex was already present
     */
    bool add_vertex(Vertex2 u, const VertexProperty& p = VertexProperty()) {
      assert(u != Vertex2());
      bool is_present = contains(u);
      data2_[u].property = p;
      return is_present;
    }

    /**
     * Adds an edge {u, v} to the graph. If the edge already exists, its
     * data is overwritten. If u or v are not present, they are added.
     * \returns the edge and bool representing if the edge was already present
     */
    std::pair<edge, bool>
    add_edge(Vertex1 u, Vertex2 v, const EdgeProperty& p = EdgeProperty()) {
      if (contains(u, v)) {
        EdgeProperty* p_ptr = data1_[u].neighbors[v];
        *p_ptr = p;
        return std::make_pair(edge(u, v, p_ptr), false);
      }
      EdgeProperty* p_ptr = new EdgeProperty(p);
      data1_[u].neighbors[v] = p_ptr;
      data2_[v].neighbors[u] = p_ptr;
      ++edge_count_;
      return std::make_pair(edge(u, v, p_ptr), true);
    }

    /**
     * Removes a type-1 vertex and all its incident edges from the graph
     */
    void remove_vertex(Vertex1 u) {
      clear_edges(u);
      data1_.erase(u);
    }

    /**
     * Removes a type-2 vertex and all its incident edges from the graph
     */
    void remove_vertex(Vertex2 u) {
      clear_edges(u);
      data2_.erase(u);
    }

    /**
     * Removes an undirected edge {u, v}. The edge must be present.
     */
    void remove_edge(Vertex1 u, Vertex2 v) {
      // find the edge (u, v)
      vertex1_data& data = find_vertex_data(u);
      typename neighbor2_map::iterator it = data.neighbors.find(v);
      assert(it != data.neighbors.end());

      // delete the edge data and the two symmetric edges
      delete it->second;
      data.neighbors.erase(it);
      data2_[v].neighbors.erase(u);
      --edge_count_;
    }

    /**
     * Removes all edges incident to a type-1 vertex
     */
    void clear_edges(Vertex1 u) {
      neighbor2_map& neighbors = find_vertex_data(u).neighbors;
      edge_count_ -= neighbors.size();
      foreach (typename neighbor2_map::reference p, neighbors) {
        data2_[p.first].neighbors.erase(u);
        delete p.second;
      }
      neighbors.clear();
    }

    /**
     * Removes all edges incident to a type-2 vertex
     */
    void clear_edges(Vertex2 u) {
      neighbor1_map& neighbors = find_vertex_data(u).neighbors;
      edge_count_ -= neighbors.size();
      foreach (typename neighbor1_map::reference p, neighbors) {
        data1_[p.first].neighbors.erase(u);
        delete p.second;
      }
      neighbors.clear();
    }

    /**
     * Removes all edges from the graph
     */
    void clear_edges() {
      free_edge_data();
      foreach(typename vertex1_data_map::reference p, data1_) {
        p.second.neighbors.clear();
      }
      foreach(typename vertex2_data_map::reference p, data2_) {
        p.second.neighbors.clear();
      }
      edge_count_ = 0;
    }

    /**
     * Removes all vertices and edges from the graph
     */
    void clear() {
      free_edge_data();
      data1_.clear();
      data2_.clear();
      edge_count_ = 0;
    }

    /**
     * Saves the graph to an archive
     */
    void save(oarchive& ar) const {
      ar << num_vertices1() << num_vertices2() << num_edges();
      foreach(typename vertex1_data_map::const_reference p, data1_) {
        ar << p.first << p.second.property;
      }
      foreach(typename vertex2_data_map::const_reference p, data2_) {
        ar << p.first << p.second.property;
      }
      foreach(edge e, edges()) {
        ar << e.v1() << e.v2() << *e.property_;
      }
    }

    /**
     * Loads the graph from an archive
     */
    void load(iarchive& ar) {
      clear();
      size_t num_vertices1, num_vertices2, num_edges;
      Vertex1 u;
      Vertex2 v;
      ar >> num_vertices1 >> num_vertices2 >> num_edges;
      while (num_vertices1-- > 0) {
        ar >> u; add_vertex(u);
        ar >> operator[](u);
      }
      while (num_vertices2-- > 0) {
        ar >> v; add_vertex(v);
        ar >> operator[](v);
      }
      while (num_edges-- > 0) {
        ar >> u >> v;
        ar >> operator[](add_edge(u, v).first);
      }
    }

    // Private helper functions
    //==========================================================================
  private: 
    const vertex1_data& find_vertex_data(Vertex1 u) const { 
      typename vertex1_data_map::const_iterator it = data1_.find(u);
      assert(it != data1_.end());
      return it->second;
    }

    const vertex2_data& find_vertex_data(Vertex2 u) const { 
      typename vertex2_data_map::const_iterator it = data2_.find(u);
      assert(it != data2_.end());
      return it->second;
    }

    vertex1_data& find_vertex_data(Vertex1 u) { 
      typename vertex1_data_map::iterator it = data1_.find(u);
      assert(it != data1_.end());
      return it->second;
    }

    vertex2_data& find_vertex_data(Vertex2 u) { 
      typename vertex2_data_map::iterator it = data2_.find(u);
      assert(it != data2_.end());
      return it->second;
    }

    void free_edge_data() {
      foreach(edge e, edges()) {
        if(e.property_) {
          delete const_cast<edge_property*>(e.property_);
        }
      }
    }

    // Implementation of edge iterators
    //==========================================================================
  public:
    class edge_iterator : 
      public std::iterator<std::forward_iterator_tag, edge> {
    public:
      typedef edge reference;
      typedef typename vertex1_data_map::const_iterator primary_iterator;
      typedef typename neighbor2_map::const_iterator secondary_iterator;

    private:
      primary_iterator it1_, end1_;
      secondary_iterator it2_;
    public:
      edge_iterator() { }
      edge_iterator(primary_iterator it1, primary_iterator end1)
        : it1_(it1), end1_(end1) {
        // skip all the empty neighbor maps
        while (it1_ != end1_ && it1_->second.neighbors.empty()) {
          ++it1_;
        }
        // if not reached the end, initialize the secondary iterator
        if (it1_ != end1_) {
          it2_ = it1_->second.neighbors.begin();
        }
      }
      edge operator*() const {
        return edge(it1_->first, it2_->first, it2_->second);
      }
      edge_iterator& operator++() {
        ++it2_;
        if (it2_ == it1_->second.neighbors.end()) {
          // at the end of the neighbor map; advance the primary iterator
          do {
            ++it1_;
          } while (it1_ != end1_ && it1_->second.neighbors.empty());
          if (it1_ != end1_) {
            it2_ = it1_->second.neighbors.begin();
          }
        }
        return *this;
      }     
      edge_iterator operator++(int) {
        edge_iterator copy = *this;
        operator++();
        return copy;
      }
      bool operator==(const edge_iterator& o) const {
        return
          (it1_ == end1_ && o.it1_ == o.end1_) ||
          (it1_ == o.it1_ && it2_ == o.it2_);
      }     
      bool operator!=(const edge_iterator& other) const {
        return !(*this == other);
      }
    }; // class edge_iterator

    template <typename Target, typename Neighbors>
    class in_edge_iterator :
      public std::iterator<std::forward_iterator_tag, edge> {
    public:
      typedef edge reference; // override the base class reference type
      typedef typename Neighbors::const_iterator iterator; // base iterator type
    private:
      Target target_;
      iterator it_;
    public:
      in_edge_iterator() { }
      in_edge_iterator(Target target, iterator it) : target_(target), it_(it) { }
      edge operator*() const { return edge(it_->first, target_, it_->second); }
      in_edge_iterator& operator++() { ++it_; return *this; }
      in_edge_iterator operator++(int) {
        in_edge_iterator copy(*this);
        ++it_;
        return copy;
      }
      bool operator==(const in_edge_iterator& o) const {
        return it_ == o.it_;
      }     
      bool operator!=(const in_edge_iterator& o) const {
        return it_ != o.it_;
      }
    }; // class in_edge_iterator

    template <typename Source, typename Neighbors>
    class out_edge_iterator :
      public std::iterator<std::forward_iterator_tag, edge> {
    public:
      typedef edge reference; // override the base class reference type
      typedef typename Neighbors::const_iterator iterator; // base iterator type
    private:
      Source source_;
      iterator it_;
    public:
      out_edge_iterator() { }
      out_edge_iterator(Source source, iterator it) : source_(source), it_(it) { }
      edge operator*() const { return edge(source_, it_->first, it_->second); }
      out_edge_iterator& operator++() { ++it_; return *this; }
      out_edge_iterator operator++(int) {
        out_edge_iterator copy(*this);
        ++it_;
        return copy;
      }
      bool operator==(const out_edge_iterator& o) const {
        return it_ == o.it_;
      }     
      bool operator!=(const out_edge_iterator& o) const {
        return it_ != o.it_;
      }
    }; // class out_edge_iterator

    // Private data
    //==========================================================================
  private:
    vertex1_data_map data1_;
    vertex2_data_map data2_;
    size_t edge_count_;
      
  }; // class bipartite_graph

} // namespace sill

#include <sill/macros_undef.hpp>

// TODO: boost graph traits

#endif
