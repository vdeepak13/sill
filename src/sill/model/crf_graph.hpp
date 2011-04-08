#ifndef SILL_CRF_GRAPH_HPP
#define SILL_CRF_GRAPH_HPP

#include <fstream>
#include <list>
#include <map>
#include <stdexcept>
#include <vector>

#include <sill/base/universe.hpp>
#include <sill/copy_ptr.hpp>
#include <sill/datastructure/set_index.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/graph/undirected_graph.hpp>
#include <sill/graph/property_functors.hpp>
#include <sill/iterator/map_value_iterator.hpp>
#include <sill/learning/dataset/datasource.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/serialization/list.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  template <typename InputVar, typename OutputFactor, typename OptVector>
  class crf_factor;

  namespace impl {

    /**
     * The information stored with each vertex of the CRF graph.
     *
     * If variable v != NULL, then this vertex is a variable vertex for v.
     *  - Y, X must be NULL.
     * If variable v == NULL, then this vertex is a factor vertex.
     *  - Y, X must store the arguments of the factor.
     *
     * @tparam OutputVariable   Type of output variable for this CRF.
     * @tparam InputVariable    Type of input variable for this CRF.
     * @tparam VertexProperty  Type of value stored at all vertices by user.
     */
    template <typename OutputVariable, typename InputVariable,
              typename VertexProperty>
    struct crf_graph_vertex_info {

      typedef OutputVariable output_variable_type;

      typedef InputVariable input_variable_type;

      typedef std::set<output_variable_type*> output_domain_type;

      typedef std::set<input_variable_type*> input_domain_type;

      //! Variable in Y (if variable vertex).
      output_variable_type* v;

      //! Output variables Y for this factor (if factor vertex).
      output_domain_type* Y;

      //! Finite input variables X for this factor (if factor vertex).
      //! This is a copy_ptr since some CRFs have the same X for every factor.
      copy_ptr<input_domain_type> X_ptr;

      //! The externally managed information.
      VertexProperty property;

      //! Default constructor. Default-initializes the property
      //! (in case it is a POD type).
      crf_graph_vertex_info()
        : v(NULL), Y(NULL), property() { }

      //! Copy constructor.
      crf_graph_vertex_info(const crf_graph_vertex_info& info)
        : v(info.v), Y(NULL), property(info.property) {
        if (info.Y) {
          Y = new output_domain_type(*(info.Y));
          X_ptr = info.X_ptr;
        }
      }

      //! Constructor for a variable vertex.
      crf_graph_vertex_info(output_variable_type* v,
                            const VertexProperty& property = VertexProperty())
        : v(v), Y(NULL), property(property) {
        assert(v != NULL);
      }

      //! Constructor for a factor vertex.
      //! This makes its own copy of Y.
      crf_graph_vertex_info(const output_domain_type* const Y,
                            copy_ptr<input_domain_type> X_ptr,
                            const VertexProperty& property = VertexProperty())
        : v(NULL), Y(NULL), X_ptr(X_ptr), property(property) {
        assert(Y && X_ptr);
        this->Y = new output_domain_type(*Y);
      }

      virtual ~crf_graph_vertex_info() {
        if (Y != NULL)
          delete(Y);
      }

      //! Serialize members.
      //! NOTE: This does NOT save the vertex property;
      //!       the caller must save that info separately.
      void save(oarchive & ar) const {
        if (v) { // variable vertex
          ar << true << v;
        } else if (Y && X_ptr) { // factor vertex
          ar << false << *Y << *X_ptr;
        } else {
          throw std::runtime_error("crf_graph_vertex_info::save called for vertex which was neither a variable vertex nor a factor vertex.");
        }
      }

      //! Deserialize members
      //! NOTE: This does NOT load the vertex property;
      //!       the caller must load that info separately.
      void load(iarchive & ar) {
        if (Y) {
          delete(Y);
          Y = NULL;
        }
        if (X_ptr)
          X_ptr.reset();
        bool var_vertex;
        ar >> var_vertex;
        if (var_vertex) {
          ar >> v;
        } else {
          Y = new output_domain_type();
          X_ptr.reset(new input_domain_type());
          ar >> *Y >> *X_ptr;
        }
      }

      //! Assignment operator.
      crf_graph_vertex_info& operator=(const crf_graph_vertex_info& info) {
        if (Y) {
          delete(Y);
          Y = NULL;
        }
        v = info.v;
        if (info.Y) {
          Y = new output_domain_type(*(info.Y));
        }
        X_ptr = info.X_ptr;
        property = info.property;
        return *this;
      }

    }; // class crf_graph_vertex_info

    template <typename ID, typename OD, typename VP>
    std::ostream& operator<<(std::ostream& out,
                             const crf_graph_vertex_info<ID, OD, VP>& info) {
      out << "(";
      if (info.v != NULL) {
        out << "v[" << info.v << "]";
      } else {
        assert(info.Y != NULL);
        out << "f[";
        print_range(out, *(info.Y), '{', ' ', '}');
        out << ", ";
        print_range(out, *(info.X_ptr), '{', ' ', '}');
        out << "]";
      }
      out << " " << info.property << ")";
      return out;
    }

  } // namespace sill::impl

  /**
   * This is the graph of a factor graph representation of a
   * conditional random field (CRF) for representing distributions P(Y | X).
   *
   * A factor graph is a bipartite graphical model where the two
   * sets of vertices correspond to variables and factors, respectively,
   * and there is an undirected edge between a variable and a factor iff the
   * variable is in the domain of the factor.
   * For CRFs, there are only variable vertices for output variables (Y),
   * not for input variables (X).
   * Any number of input variables may be associated with each factor.
   *
   * @tparam OutputVariable   Type of output variable for this CRF.
   * @tparam InputVariable    Type of input variable for this CRF.
   * @tparam Variable         Type of variable for this CRF (from which
   *                          both InputVariable and OutputVariable inherit).
   * @tparam VertexProperty a type that models the DefaultConstructible
   *         and the CopyConstructible concept.
   *
   * @see crf_model
   * \ingroup model
   */
  template <typename OutputVariable, typename InputVariable, typename Variable,
            typename VertexProperty = void_>
  class crf_graph {

    concept_assert((DefaultConstructible<VertexProperty>));

    // Private and public type declarations
    // =========================================================================
  public:

    //! Type of variables in Y.
    typedef OutputVariable output_variable_type;

    //! Type of variables in X.
    typedef InputVariable input_variable_type;

    //! Type of variables in both Y,X.
    typedef Variable variable_type;

    //! Type of output domain Y.
    typedef std::set<output_variable_type*> output_domain_type;

    //! Type of input domain X.
    typedef std::set<input_variable_type*> input_domain_type;

    //! Type of domain for both Y,X.
    typedef std::set<variable_type*> domain_type;

  private:

    //! Information associated with each vertex.
    typedef impl::crf_graph_vertex_info<output_variable_type,
                                        input_variable_type,
                                        VertexProperty> vertex_info;

  public:

    //! The underlying graph type (needs to be public for SWIG)
    typedef undirected_graph<size_t, vertex_info, void_> graph_type;

    // Graph types
    // (We use the specific types here, so that we do not have manually
    //  instantiate the graph_type template in SWIG.)
    typedef typename graph_type::vertex          vertex;
    typedef typename graph_type::edge            edge;
    typedef VertexProperty vertex_property;

    // Graph iterators
    typedef typename graph_type::vertex_iterator    vertex_iterator;
    typedef typename graph_type::neighbor_iterator  neighbor_iterator;
    typedef typename graph_type::edge_iterator      edge_iterator;
    typedef typename graph_type::in_edge_iterator   in_edge_iterator;
    typedef typename graph_type::out_edge_iterator  out_edge_iterator;
    typedef map_value_iterator<std::map<output_variable_type*, vertex> >
      variable_vertex_iterator;

    /**
     * Iterator for all vertices which may be reached from a given vertex v
     * in exactly 2 hops, not counting v.
     */
    class neighbor2_iterator
      : public std::iterator<std::forward_iterator_tag, const vertex> {

      //! The crf_graph
      const crf_graph* const graph;

      //! The given vertex
      vertex v;

      //! Iterators over neighbors of v
      neighbor_iterator neighbor_it, neighbor_end;

      //! Iterators over neighbors of *neighbor_it.
      neighbor_iterator neighbor2_it, neighbor2_end;

      //! True if end iterator.
      bool end_iter;

      //! Temp null vertex.
      vertex tmp_null_vertex;

      //! If the current vertex is valid, return.
      //! Otherwise, iterate until it is valid.
      void iterate_until_valid() {
        while(neighbor2_it == neighbor2_end ||
              *neighbor2_it == v) {
          while(neighbor2_it == neighbor2_end) {
            ++neighbor_it;
            if (neighbor_it == neighbor_end) {
              end_iter = true;
              return;
            }
            boost::tie(neighbor2_it, neighbor2_end) =
              graph->neighbors(*neighbor_it);
          }
          if (*neighbor2_it == v)
            ++neighbor2_it;
        }
      }

    public:

      //! End iterator constructor
      neighbor2_iterator() : graph(NULL), end_iter(true) { }

      //! Constructor for an iterator for vertex v's 2-neighbors.
      neighbor2_iterator(const crf_graph* const graph, vertex v)
        : graph(graph), v(v), end_iter(false) {
        boost::tie(neighbor_it, neighbor_end) = graph->neighbors(v);
        if (neighbor_it == neighbor_end)
          end_iter = true;
        boost::tie(neighbor2_it, neighbor2_end) =
          graph->neighbors(*neighbor_it);
        iterate_until_valid();
      }

      //! Prefix increment.
      neighbor2_iterator& operator++() {
        if (end_iter)
          return *this;
        ++neighbor2_it;
        iterate_until_valid();
        return *this;
      }

      //! Postfix increment.
      neighbor2_iterator operator++(int) {
        neighbor2_iterator tmp(*this);
        ++(*this);
        return tmp;
      }

      //! Returns a const reference to the current vertex.
      const vertex& operator*() const {
        if (end_iter)
          return tmp_null_vertex;
        else
          return *neighbor2_it;
      }

      //! Returns truth if the two neighbor2 iterators are the same.
      bool operator==(const neighbor2_iterator& it) const {
        if (end_iter && it.end_iter)
          return true;
        return (graph == it.graph && v == it.v &&
                neighbor_it == it.neighbor_it &&
                neighbor2_it == it.neighbor2_it && end_iter == it.end_iter);
      }

      //! Returns truth if the two neighbor2 iterators are different.
      bool operator!=(const neighbor2_iterator& it) const {
        return !operator==(it);
      }

    }; // class neighbor2_iterator

    // Constructors and destructors
    //==========================================================================

    /**
     * Creates a CRF graph with no factors and no variables.
     * Use the add_factor method to add factors and variables.
     */
    crf_graph() : next_vertex(0) { }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << variable_index_ << factor_vertices_ << graph << next_vertex
         << Y_ << X_ << X_counts;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> variable_index_ >> factor_vertices_ >> graph >> next_vertex
         >> Y_ >> X_ >> X_counts;
    }

    // Graph accessors
    // =========================================================================

    //! Returns an ordered set of all vertices.
    std::pair<vertex_iterator, vertex_iterator>
    vertices() const {
      return graph.vertices();
    }

    //! Returns an ordered set of all variable vertices.
    std::pair<variable_vertex_iterator, variable_vertex_iterator>
    variable_vertices() const {
      return std::make_pair
        (variable_vertex_iterator(variable_index_.begin()),
         variable_vertex_iterator(variable_index_.end()));
    }

    //! Returns an ordered set of all factor vertices.
    const std::list<vertex>& factor_vertices() const {
      return factor_vertices_;
    }

    //! @return  0 if variable vertex; 1 if factor vertex; 2 if bad vertex
    size_t vertex_type(const vertex& v) const {
      if (graph[v].v != NULL)
        return 0;
      else if (graph[v].Y != NULL)
        return 1;
      else
        return 2;
    }

    //! Returns the vertices adjacent to u
    std::pair<neighbor_iterator, neighbor_iterator>
    neighbors(const vertex& u) const {
      return graph.neighbors(u);
    }

    //! Returns the vertices adjacent to variable u.
    std::pair<neighbor_iterator, neighbor_iterator>
    neighbors(output_variable_type* u) const {
      return graph.neighbors(variable2vertex(u));
    }

    //! Returns the vertices at distance 2 from u.
    //! (For variable vertices, this returns variable vertices sharing factors;
    //!  for factor vertices, this returns factor vertices sharing variables.)
    //! Note: This assumes that the graph is bipartite, which is true for
    //!       factor graphs.
    std::pair<neighbor2_iterator, neighbor2_iterator>
    neighbors2(const vertex& u) const {
      return std::make_pair(neighbor2_iterator(this, u), neighbor2_iterator());
    }

    //! Returns the vertices at distance 2 from u's vertex.
    //! I.e., this returns the neighboring variables.
    std::pair<neighbor2_iterator, neighbor2_iterator>
    neighbors2(output_variable_type* u) const {
      return neighbors2(variable2vertex(u));
    }

    //! Returns true if u and v are neighbors.
    bool are_neighbors(output_variable_type* u, output_variable_type* v) const {
      vertex v_vert(variable2vertex(v));
      foreach(const vertex& t, neighbors2(u))
        if (t == v_vert)
          return true;
      return false;
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

    //! Returns the number of edges adjacent to a variable's vertex
    size_t degree(output_variable_type* u) const {
      return graph.degree(variable2vertex(u));
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

    /*
    //! Returns the property associated with an edge
    edge_property& operator[](const edge& e) {
      return graph[e].property;
    }
    */

    //! Returns the property associated with a vertex
    const vertex_property& operator[](const vertex& u) const {
      return graph[u].property;
    }

    /*
    //! Returns the property associated with an edge
    const edge_property& operator[](const edge& e) const {
      return graph[e].property;
    }
    */

    //! Returns a view of all vertex properties.
    //! This crf_graph must outlive the returned view.
    forward_range<const vertex_property&> vertex_properties() const {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    /*
    //! Returns a view of all edge properties
    //! This crf_graph must outlive the returned view.
    forward_range<const edge_property&> edge_properties() const {
      return make_transformed(edges(), edge_property_functor(*this));
    }
    */

    //! Returns a null vertex
    static vertex null_vertex() {
      return graph_type::null_vertex();
    }

    // Getters and helpers
    //==========================================================================

    //! The number of factors in this graphical model
    size_t size() const {
      return factor_vertices_.size();
    }

    //! The number of arguments (variables) in this model.
    size_t num_arguments() const {
      return Y_.size() + X_.size();
    }

    //! Returns all arguments (variables) in this model (Y,X).
    //! Note: It is more efficient to get Y,X separately.
    domain_type arguments() const {
      domain_type args(Y_.begin(), Y_.end());
      args.insert(X_.begin(), X_.end());
      return args;
    }

    //! Returns all output variables Y for this model.
    const output_domain_type& output_arguments() const {
      return Y_;
    }

    //! Returns all input variables X for this model.
    const input_domain_type& input_arguments() const {
      return X_;
    }

    //! Given a factor vertex, returns all arguments for the factor.
    //! If the vertex is a variable vertex, returns an empty set.
    //! Note: It is more efficient to get Y,X separately.
    domain_type arguments(vertex v) const {
      if (graph[v].Y == NULL)
        return domain_type();
      domain_type args;
      args.insert(graph[v].Y->begin(), graph[v].Y->end());
      args.insert(graph[v].X_ptr->begin(), graph[v].X_ptr->end());
      return args;
    }

    //! Given a factor vertex, returns all output variables for the factor.
    //! If the vertex is a variable vertex, asserts false.
    const output_domain_type& output_arguments(vertex v) const {
      assert(graph[v].Y != NULL);
      return *(graph[v].Y);
    }

    //! Given a factor vertex, returns all input variables for the factor.
    //! If the vertex is a variable vertex, asserts false.
    const input_domain_type& input_arguments(vertex v) const {
      assert(graph[v].X_ptr);
      return *(graph[v].X_ptr);
    }

    //! Given a factor vertex, returns all input variables for the factor.
    //! If the vertex is a variable vertex, asserts false.
    copy_ptr<input_domain_type> input_arguments_ptr(vertex v) const {
      assert(graph[v].X_ptr);
      return graph[v].X_ptr;
    }

    //! Given a variable, return its variable vertex.
    //! Returns null_vertex() if the variable is not present in the CRF.
    vertex variable2vertex(output_variable_type* v) const {
      typename std::map<output_variable_type*, vertex>::const_iterator
        it(variable_index_.find(v));
      if (it == variable_index_.end())
        return null_vertex();
      else
        return it->second;
    }

    //! Given a variable vertex, return its variable.
    //! Returns NULL if the vertex is not a variable vertex.
    //! Fails if the vertex is not present in the graph.
    output_variable_type* vertex2variable(vertex v) const {
      return graph[v].v;
    }

    // Mutating methods
    //==========================================================================

    /*
     * Add a factor to this factor graph. All the variables are added
     * to this graphical model, potentially changing the domain.
     * Note: This does not allow constant factors.
     * Note: Use the other add_factor() if your CRF has many factors with
     *       the same input variables.
     */
    vertex add_factor(const output_domain_type& Yvars,
                      const input_domain_type& Xvars,
                      const VertexProperty& property = VertexProperty()) {
      return add_factor
        (Yvars, copy_ptr<input_domain_type>(new input_domain_type(Xvars)),
         property);
    }

    /**
     * Add a factor to this factor graph. All the variables are added
     * to this graphical model, potentially changing the domain.
     * Note: This does not allow constant factors.
     */
    vertex add_factor(const output_domain_type& Yvars,
                      copy_ptr<input_domain_type> Xvars,
                      const VertexProperty& property = VertexProperty()) {
      assert(Xvars);
      vertex vert(next_vertex++);
      graph.add_vertex(vert, vertex_info(&Yvars, Xvars, property));
      // Add variable vertices as necessary.
      std::vector<vertex> newY(unsafe_add_Y(Yvars));
      // Connect factor vertex to variable vertices.
      foreach(output_variable_type* yv, Yvars) {
        graph.add_edge(vert, variable_index_[yv]);
      }
      // Update metadata.
      factor_vertices_.push_back(vert);
      X_.insert(Xvars->begin(), Xvars->end());
      foreach(input_variable_type* xv, *Xvars) {
        typename std::map<input_variable_type*, size_t>::iterator
          X_counts_it(X_counts.find(xv));
        if (X_counts_it == X_counts.end())
          X_counts[xv] = 1;
        else
          ++(X_counts_it->second);
      }
      // Check to make sure Y,X stay separate.
      if (!set_disjoint(Y_, *Xvars) || !set_disjoint(X_, Yvars)) {
        throw std::invalid_argument
          (std::string("crf_graph::add_factor() given overlapping Y,X domains:")
           + "\nY: " + to_string(Y_) + "\nX: " + to_string(X_) + "\n");
      }
      return vert;
    }

    //! Removes a factor vertex from the CRF graph, removing the factor
    //! arguments as well if no other factors use them.
    void remove_factor(const vertex& u) {
      assert(vertex_type(u) == 1);
      foreach(input_variable_type* x, input_arguments(u)) {
        size_t& cnt = X_counts[x];
        if (cnt == 1) {
          X_counts.erase(x);
          X_.erase(x);
        } else {
          --cnt;
        }
      }
      std::list<vertex> u_arg_vertices; 
      foreach(const vertex& v, neighbors(u))
        u_arg_vertices.push_back(v);
      graph.remove_vertex(u);
      foreach(const vertex& v, u_arg_vertices) {
        if (degree(v) == 0) {
          output_variable_type* vvar = vertex2variable(v);
          variable_index_.erase(vvar);
          Y_.erase(vvar);
        }
      }
      for (typename std::list<vertex>::iterator it(factor_vertices_.begin());
           it != factor_vertices_.end(); ++it) {
        if (*it == u) {
          factor_vertices_.erase(it);
          break;
        }
      }
    }

    //! Clears all factors and variables from this graph.
    virtual void clear() {
      variable_index_.clear();
      factor_vertices_.clear();
      graph.clear();
      next_vertex = 0;
      Y_.clear();
      X_.clear();
      X_counts.clear();
    }

    //! Simplifies the model by removing unary factors (over a single Y)
    //! if another factor contains that Y variable.
    //! @todo Replace this with simplify() from factor_graph_model.
    //! @todo This should also check to see if there are 2 unary factors
    //!       for a variable.
    virtual void simplify_unary() {
      simplify_unary_helper();
    }

    // Below are methods I could copy from factor_graph_model:
    //    void randomly_shuffle_neighbors(){
    //    virtual void simplify() {
    //    virtual void simplify_stable() {
    //    void normalize() {
    //    virtual size_t work_per_update(const vertex_type& v) const {
    //    virtual bool is_consistent(){
    //    void print_degree_distribution() {
    //    virtual void print_adjacency(std::ostream& out) const {
    //    virtual void print_vertex_info(std::ostream& out) const {
    //    vertex_id_type vertex2id(const vertex_type& v) const{
    //    vertex_type id2vertex(vertex_id_type id) const{
    //    void refill_vertex2id(){

    /////////////////////////////////////////////////////////////////
    // factorized_model<F> interface
    /////////////////////////////////////////////////////////////////

    //! Prints the arguments of the model.
    template <typename OutputStream>
    void print(OutputStream& out) const {
      out << "Y: " << output_arguments() << "\n";
      out << "X: " << input_arguments() << "\n";
      out << graph;
      /*
      foreach(const vertex& v, factor_vertices()) {
        out << "f[" << output_arguments(v) << "; "
            << input_arguments(v) << "]" << "\n";
      }
      */
    }

    virtual operator std::string() const {
      std::ostringstream out;
      print(out);
      return out.str();
    }

    /**
     * Print the structure to a file in a format which can be read in again
     * later.  The format is as follows:
     *  - Each line represents one factor by a set of Y indices indexed from 0,
     *    followed by a semicolon and a set of X indices.
     * @param ds      The format is defined w.r.t. the variable ordering from
     *                this datasource.
     * @param save_x  If false, do not have semicolon or X indices.
     *                (default = true)
     */
    void
    save_structure(const std::string& filename, const datasource& ds,
                   bool save_x = true) const {
      std::ofstream fout(filename.c_str());
      assert(fout.good());
      std::map<variable*, size_t> var_order_map(ds.variable_order_map());
      foreach(const vertex& v, factor_vertices()) {
        foreach(output_variable_type* y, output_arguments(v)) {
          fout << var_order_map[y] << "\t";
//          fout << ds.var_order_index(y) << "\t";
        }
        if (save_x) {
          fout << ";\t";
          foreach(input_variable_type* x, input_arguments(v)) {
            fout << var_order_map[x] << "\t";
//            fout << ds.var_order_index(x) << "\t";
          }
        }
        fout << "\n";
      }
      fout.flush();
      fout.close();
    }

    /**
     * Loads the structure from a file produced by save_structure().
     * This erases the current graph.
     * @param ds  The format is defined w.r.t. the variable ordering from this
     *            datasource, so the structure must have been saved using an
     *            analogous datasource.
     * @param suppress_warnings  If true, do not print out warning messages.
     */
    void
    load_structure(const std::string& filename, const datasource& ds,
                   bool suppress_warnings = false) {
      clear();
      std::ifstream fin(filename.c_str());
      if (!fin.good()) {
        throw std::runtime_error
          ("crf_graph::load_structure() was unable to open file: " + filename);
      }
      std::string tmpstring;
      std::istringstream is;
      size_t ind;
      var_vector var_order(ds.var_order());
      while (fin.good()) {
        getline(fin, tmpstring);
        is.clear();
        is.str(tmpstring);
        output_domain_type tmpY;
        while (is >> ind) { // Read in outputs Y
          assert(ind < var_order.size());
          tmpY.insert(dynamic_cast<output_variable_type*>(var_order[ind]));
        }
        // Assume the unreadable item was the semicolon.
        copy_ptr<input_domain_type> tmpX_ptr(new input_domain_type());
        ind = tmpstring.find_first_of(";");
        if (ind != std::string::npos) { // otherwise, no input variables on line
          ++ind;
          assert(ind < tmpstring.size());
          tmpstring = tmpstring.substr(ind);
          is.clear();
          is.str(tmpstring);
          while (is >> ind) { // Read in inputs X
            assert(ind < var_order.size());
            tmpX_ptr->insert
              (dynamic_cast<input_variable_type*>(var_order[ind]));
          }
        }
        assert(set_disjoint(tmpY, *tmpX_ptr));
        if (tmpY.size() != 0) {
          add_factor(tmpY, tmpX_ptr);
        } else {
          if ((tmpstring.size() != 0) && (!suppress_warnings)) {
            std::cerr << "crf_graph::load_structure read in a factor with no Y"
                      << " variables (and skipped it) on this line:\n"
                      << tmpstring << std::endl;
          }
        }
      }
      fin.close();
    } // load_structure

//     /////////////////////////////////////////////////////////////////
//     // graphical_model<F> interface
//     /////////////////////////////////////////////////////////////////

//     /**
//      * Returns a minimal markov graph that captures the dependencies
//      * in this model
//      * \todo implement this
//      */
//      sill::markov_graph<variable_type> markov_graph() {
//       assert(false); // TODO
//       return sill::markov_graph<variable_type>();
//     }

//     /**
//      * Determines whether x and y are separated, given z
//      * \todo implement this
//      */
//     bool d_separated(const domain_type& x, const domain_type& y,
//                      const domain_type& z = domain_type::empty_set) {
//       assert(false); // TODO
//       return false;
//     }

    // Unsafe Mutating Methods
    // These could be moved to a child class.
    //==========================================================================

    //! Add the variables in Y not already present, without any factors.
    //! @return  vertices for variables which were added
    std::vector<vertex>
    unsafe_add_Y(const output_domain_type& Yvars,
                 const VertexProperty& property = VertexProperty()) {
      std::vector<vertex> new_vertices;
      foreach(output_variable_type* yvar, Yvars) {
        if (Y_.count(yvar) != 0)
          continue;
        vertex v(next_vertex++);
        new_vertices.push_back(v);
        graph.add_vertex(v, vertex_info(yvar, property));
        variable_index_[yvar] = v;
        Y_.insert(yvar);
      }
      return new_vertices;
    }

  protected:

    /**
     * Add a factor to this factor graph. All the variables are added
     * to this graphical model, potentially changing the domain.
     * Note: This does not allow constant factors.
     */
    vertex
    add_factor_no_check(const output_domain_type& Yvars,
                        copy_ptr<input_domain_type> Xvars,
                        const VertexProperty& property = VertexProperty()) {
      assert(Xvars);
      vertex vert(next_vertex++);
      graph.add_vertex(vert, vertex_info(&Yvars, Xvars, property));
      // Add variable vertices as necessary.
      std::vector<vertex> newY(unsafe_add_Y(Yvars));
      // Connect factor vertex to variable vertices.
      foreach(output_variable_type* yv, Yvars) {
        graph.add_edge(vert, variable_index_[yv]);
      }
      // Update metadata.
      factor_vertices_.push_back(vert);
      X_.insert(Xvars->begin(), Xvars->end());
      foreach(input_variable_type* xv, *Xvars) {
        typename std::map<input_variable_type*, size_t>::iterator
          X_counts_it(X_counts.find(xv));
        if (X_counts_it == X_counts.end())
          X_counts[xv] = 1;
        else
          ++(X_counts_it->second);
      }
      return vert;
    } // add_factor_nocheck

    // Protected data members and methods
    // =========================================================================
  protected:

    //! A map from output variables to their variable vertices.
    std::map<output_variable_type*, vertex> variable_index_;

    //! List of factor vertices.
    std::list<vertex> factor_vertices_;

    //! The underlying graph.
    graph_type graph;

    //! The next vertex id
    size_t next_vertex;

    //! Output variables Y
    output_domain_type Y_;

    //! Input variables X
    input_domain_type X_;

    //! Mapping: input variable x --> number of factors using x
    //! This is needed to make remove_factor() efficient.
    std::map<input_variable_type*, size_t> X_counts;

    //! Removes unnecessary unary vertices.
    //! @return  VertexProperty values for removed vertices.
    std::list<VertexProperty> simplify_unary_helper() {
      std::list<VertexProperty> removed_properties;
      std::set<vertex> remove_vertices_;
      foreach(const vertex& u, factor_vertices_) {
        if (output_arguments(u).size() == 1) {
          foreach(const vertex& v, neighbors2(u)) {
            if ((output_arguments(v).size() > 1) &&
                includes(input_arguments(v), input_arguments(u))) {
              remove_vertices_.insert(u);
              break;
            }
          }
        }
      }
      foreach(const vertex& u, remove_vertices_) {
        removed_properties.push_back(graph[u].property);
        graph.remove_vertex(u);
      }
      for (typename std::list<vertex>::iterator it(factor_vertices_.begin());
           it != factor_vertices_.end(); ) {
        if (remove_vertices_.count(*it) != 0) {
          typename std::list<vertex>::iterator it2(it);
          ++it;
          factor_vertices_.erase(it2);
        } else {
          ++it;
        }
      }
      return removed_properties;
    } // simplify_unary_helper

  }; // crf_graph

  /**
   * Write the CRF factor graph to an output stream.  The display format is:
   *   <factor1 args>
   *   <factor2 args>
   *      ...
   *   <factorM args>
   *
   * \todo We should improve the quality of the output of this function.
   */
  template<typename OD, typename ID, typename D, typename VP>
  std::ostream& operator<<(std::ostream& out, const crf_graph<OD,ID,D,VP> crf) {
    crf.print(out);
    return out;
  } // end of operator<<

}

#include <sill/macros_undef.hpp>

#endif // SILL_CRF_GRAPH_HPP
