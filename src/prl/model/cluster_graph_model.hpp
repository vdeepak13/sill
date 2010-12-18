
#ifndef SILL_CLUSTER_GRAPH_MODEL_HPP
#define SILL_CLUSTER_GRAPH_MODEL_HPP

#include <vector>

#include <boost/range/iterator_range.hpp>

#include <sill/range/algorithm.hpp>
#include <sill/variable.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/factor.hpp>
#include <sill/factor/arguments_functor.hpp>
#include <sill/graph/tree_traversal.hpp>

#include <sill/model/interfaces.hpp>
#include <sill/model/cluster_graph.hpp>
#include <sill/inference/variable_elimination.hpp>

#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

///////////////////////////////////////////////////////////////////
// This file needs to be cleaned up & rewritten to use new graphs
// See decomposable.hpp
///////////////////////////////////////////////////////////////////

namespace sill {

  // #define SILL_VERBOSE // for debugging

  /**
   * A cluster graph representation of a probability distribution.
   * Conceptually, cluster graph model is a cluster graph, in which
   * each vertex and each edge is associated with a factor of type F.
   *
   * Note: to ensure the stability of descriptors and iterators
   *       after copying, use the supplied custom copy constructor.
   *
   * @tparam F a type that models the DistributionFactor concept
   *
   * \ingroup model
   */
  template <typename F>
  class cluster_graph_model 
    : public graphical_model<F>, protected cluster_graph<variable_h, F, F> {

#ifndef SWIG
    concept_assert((DistributionFactor<F>));
#endif

  /////////////////////////////////////////////////////////////////
  // Type declarations and member variables
  /////////////////////////////////////////////////////////////////
  public:
    //! The type of factors used in this cluster graph model
    typedef F factor_type;

    //! The cluster graph, used to represent the cluster graph model.
    typedef cluster_graph<variable*, F, F> base;
    // we _have_ to use variable* here, rather than variable_h, or else
    // SWIG does not recognize the using base::... declarations below

  #ifndef SWIG
    // Bring types from the protected parent
    // descriptors
    typedef typename base::vertex_descriptor vertex_descriptor;
    typedef typename base::edge_descriptor edge_descriptor;
    typedef typename base::vertex_pair vertex_pair;

    // iterators
    typedef typename base::vertex_iterator vertex_iterator;
    typedef typename base::edge_iterator edge_iterator;
    typedef typename base::in_edge_iterator in_edge_iterator;
    typedef typename base::out_edge_iterator out_edge_iterator;
    typedef typename base::adjacency_iterator adjacency_iterator;
    typedef typename base::inv_adjacency_iterator inv_adjacency_iterator;

    // graph category
    typedef typename base::directed_category directed_category;
    typedef typename base::edge_parallel_category edge_parallel_category;
    typedef typename base::traversal_category traversal_category;
    class graph_tag { };

    // properties
    typedef F vertex_property_type;
    typedef F edge_property_type;

    // size types TODO: can we just use size_t?
    typedef typename base::vertices_size_type vertices_size_type;
    typedef typename base::edges_size_type edges_size_type;
    typedef typename base::degree_size_type degree_size_type;

    // property maps
    typedef typename base::vertex_index_map vertex_index_map;
  #endif

    /// Bring the "safe" functions from the protected parent class
    using base::null_vertex;
    using base::vertices;
    using base::edges;
    using base::adjacent_vertices;
    using base::inv_adjacent_vertices;
    using base::out_edges;
    using base::in_edges;
    using base::source;
    using base::target;
    using base::out_degree;
    using base::in_degree;
    using base::num_vertices;
    using base::num_edges;
    using base::empty;
    using base::vertex;
    using base::edge;
    using base::reverse;

    // property maps
    using base::vertex_index;

    // cluster_graph functions
    using base::clique;
    using base::find_clique_cover;
    using base::find_clique_meets;
    using base::find_separator_cover;
    using base::find_intersecting_cliques;
    using base::treewidth;

    //! Marks that BGL accessor functions should be enabled for this class
    struct bgl { };

  protected:

    // cluster_graph property maps
    using base::clique_index;

    //! The argument variables of this distribution.
    domain args;

  /////////////////////////////////////////////////////////////////
  // Helper functions
  /////////////////////////////////////////////////////////////////
  protected:
    //! Returns a reference to the clique potential associated with a vertex.
    factor_type& potential(vertex_descriptor v) { return base::operator[](v); }

    //! Returns a reference to the separator potential associated with an edge
    factor_type& potential(edge_descriptor e) { return base::operator[](e); }

    /**
     * Functor which creates a constant c potential for a given clique.
     * \todo This will work with table factors, but possibly not others.
     *       Should we have a standard set of constructors for factors
     *       which includes factor(node_set, default_value)?
     */
    struct constant_potential_functor {
    private:
      double c;
    public:
      factor_type operator()(domain d) {
        return factor_type(d,c);
      }
      constant_potential_functor(double c) : c(c) { }
    };

  /////////////////////////////////////////////////////////////////
  // Constructors and accessors
  /////////////////////////////////////////////////////////////////
  public:
    /**
     * Default constructor. The distribution has no arguments and
     * is identically one.
     */
    cluster_graph_model() { }

    /**
     * Initializes the cluster graph model to the given set of clique marginals.
     * This creates a cluster graph which is a Bethe approximation (i.e. a
     * factor graph).
     */
    template <typename FactorRange>
    cluster_graph_model(const FactorRange& factors)
      : base(factors | sill::transformed(arguments_functor()), 
             boost::begin(factors),
             constant_potential_functor(1)) {
      // Note the base constructor adds a constant 1 potential for all the
      //  variable nodes.
      foreach(vertex_descriptor v, vertices()) args += clique(v);
    }

    //! Swaps two cluster graph models.
    void swap(cluster_graph_model& other) {
      base::swap(other);
      args.swap(other.args);
    }

    /////////////////////////////////////////////////////////////////
    // Accessors
    /////////////////////////////////////////////////////////////////
    //! Returns a reference to the clique potential associated with a vertex.
    const F& potential(vertex_descriptor v) const { return base::operator[](v);}
    const F& operator[](vertex_descriptor v) const{ return base::operator[](v);}

    //! Returns the clique potentials of this cluster graph model
    std::vector<F> clique_potentials() const {
      std::vector<F> result(num_vertices());
      size_t i = 0;
      foreach(vertex_descriptor v, vertices()) result[i++] = potential(v);
      return result;
    }

    /////////////////////////////////////////////////////////////////
    // graphical_model<NodeF> interface
    /////////////////////////////////////////////////////////////////

    domain arguments() const { return args; }

    double log_likelihood(const assignment& a) const {
      return log_likelihood(a, exp(double(1)));
    }

    logarithmic<double> operator()(const assignment& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    forward_range<const F&> factors() const {
      assert(false); // not implemented yet
      return boost::none;
    }

    //! Convert this model to a Markov graph.
    sill::markov_graph<> markov_graph() const {
      sill::markov_graph<> mg(arguments());
      foreach(vertex_descriptor v, vertices()) 
        mg.add_clique(clique(v));
      return mg;
    }

    bool d_separated(const domain& x, const domain& y,
                     const domain& z = domain::empty_set) const {
      assert(false); // not implemented yet
      return true;
    }

    /////////////////////////////////////////////////////////////////
    // Restructuring operations
    /////////////////////////////////////////////////////////////////
    /**
     * Restructures this cluster graph model so that it includes the
     * supplied cliques.  These cliques can include new variables
     * (which are not current arguments).  This adds constant 1 potentials
     * for the new vertices.
     *
     * Note: this operation invalidates all iterators and descriptors.
     *
     * @param cliques An input range over the cliques which should be
     *                added to the model.
     * \todo test this code
     */
    template <typename CliqueRange>
    void add_cliques(const CliqueRange& cliques) {
      concept_assert((InputRangeConvertible<CliqueRange, domain>));

      // Compute:
      //   new_args: the new set of arguments (to replace args)
      //   new_vars: the new variables in new_args
      //   old_vars: the old variables in new_args
      domain new_args;
      new_args.insert(cliques);
      domain new_vars = new_args.minus(args);
      domain old_vars = new_args.minus(new_vars);
      new_args.insert(args);

      // Create variable nodes for the new variables.
      std::map<variable_h, vertex_descriptor> n2v;
      foreach(variable_h v, new_vars)
        n2v[v] = base::add_vertex(domain(v), factor_type(domain(v),1));

      // Find the nodes to use for the old variables.
      foreach(variable_h n, old_vars) {
        vertex_descriptor v = find_clique_cover(n);
        assert(v != null_vertex());
        n2v[n] = v;
      }

      // For each new clique,
      foreach(domain c, cliques) {
        // Create a node for the clique
        vertex_descriptor cv = base::add_vertex(c, factor_type(c,1));
        // For each variable in the clique,
        foreach(variable_h n, c) {
          // Attach the clique to the node for variable n
          base::add_edge(cv, n2v[n]);
        }
      }

      args.swap(new_args);
    }

    /**
     * Restructures this cluster graph model so that it has a clique
     * that covers the supplied variables, and returns the vertex
     * associated with this clique.  The caller should first check
     * that no such cover exists by calling #find_clique_cover on the
     * cluster graph.
     *
     * @param vars
     *        The set of variables for which a cover should be
     *        created.  This set of variables should be a subset of
     *        the current arguments of this cluster graph model.
     */
    vertex_descriptor make_cover(const domain& vars) {
      add_cliques(boost::make_iterator_range(&vars, &vars+1));
      // Now find a cover in the new cluster graph.
      return find_clique_cover(vars);
    }

    /**
     * Merges two vertices in the cluster graph.  This operation
     * swings all edges from the source of the supplied edge to the
     * target.  The source is removed from the graph.
     */
    void merge(edge_descriptor e) {
      vertex_descriptor u = source(e);
      vertex_descriptor v = target(e);

      // Get the cliques and potentials incident to the edge.  Store
      // them by value, because altering the cluster graph below will
      // remove their storage.
      F potential_u = potential(u);

      // Update the cluster graph. This operation removes u and retains v
      merge(e);
      potential(v) *= potential_u;
    }

    /**
     * Removes a vertex from the cluster graph if it is nonmaximal with respect
     * to its neighbors.
     * If a factor is associated with the vertex, the factor is multiplied into
     * the factor of the subsuming neighbor.
     * @return true if a subsuming neighbor was found.
     * \todo Change this so that it finds non-neighbor subsuming vertices as
     *       well?
     */
    bool remove_if_nonmaximal(vertex_descriptor u) {
      // Look for a neighbor of this vertex whose clique subsumes this
      // vertex's clique.
      foreach(edge_descriptor e, out_edges(u)) {
        vertex_descriptor v = target(e);
        if (clique(v).superset_of(clique(u))) {
          potential(v) *= potential(u);
          merge(e);
          return true;
        }
      }
      return false;
    }

    /////////////////////////////////////////////////////////////////
    // Distribution operations
    /////////////////////////////////////////////////////////////////
    /**
     * Multiplies the supplied collection of factors into this
     * cluster graph model.
     *
     * @param factors A readable forward range of objects of type F
     */
    template <typename Range>
    cluster_graph_model& operator*=(const Range& factors) {
      concept_assert((ReadableForwardRangeConvertible<Range, F>));

      // Retriangulate the model so that it contains a clique for each factor.
      add_cliques(factors | sill::transformed(arguments_functor()));

      // For each factor, multiply it into a clique that subsumes it.
      // We do not use F for iteration since Range may be over a different
      // factor type that is merely convertible to F.
      foreach(const typename Range::value_type& factor, factors) {
        vertex_descriptor v = find_clique_cover(factor.arguments());
        assert(v != null_vertex());
        potential(v) *= factor;
      }
      return *this;
    }

    /**
     * Multiplies the supplied factor into this cluster graph model and
     * renormalizes the model.
     */
    cluster_graph_model& operator*=(const F& factor) {
      return (*this) *= boost::make_iterator_range(&factor, &factor + 1);
    }

    /**
     * Conditions this cluster graph model on an assignment to one or
     * more of its variables. This is a mutable operation.
     * Note this does not normalize the distribution.
     *
     * @param assignment
     *        An assignment to some variables.  This assignment is
     *        instantiated in each clique and separator factor.
     */
    cluster_graph_model& condition(const assignment& assignment) {
      // Compute the variables that are conditioned on.
      domain restricted_vars = assignment.keys();
      if (restricted_vars.disjoint_from(arguments()))
        return *this;

      // Find all cliques that contain an old variable.
      typename std::vector<vertex_descriptor> vertices;
      find_intersecting_cliques(restricted_vars, std::back_inserter(vertices));

      // Update each affected clique
      foreach(vertex_descriptor v, vertices) {
//        potential(v) = potential(v).restrict(assignment).normalize();
        potential(v) = potential(v).restrict(assignment);
        base::set_clique(v, clique(v) - restricted_vars);
      }

      // Update the arguments & recalibrate.
      args.remove(restricted_vars);
//      calibrate();
      return *this;
    }

    /**
     * Renames (a subset of) the arguments of this factor.
     *
     * @param map
     *        an object such that map[v] maps the variable handle v
     *        a type compatible variable handle; this mapping must be 1:1.
     */
    void subst_args(const var_map& var_map) {
      // Compute the variables to be replaced.
      domain old_vars = var_map.keys();

      // Find all cliques that contain an old variable.
      std::vector<vertex_descriptor> vertices;
      find_intersecting_cliques(old_vars, std::back_inserter(vertices));

      // Update affected cliques and incident separator marginals
      foreach(vertex_descriptor v, vertices) {
        base::set_clique(v, subst_vars(clique(v), var_map));
        potential(v).subst_args(var_map);
        foreach(edge_descriptor e, out_edges(v))
          potential(e).subst_args(var_map);
      }

      // Update the arguments.
      args = subst_vars(args, var_map);
    }

    /**
     * Applies the supplied functor to all clique potentials of this
     * cluster graph model.
     */
    template <typename Functor>
    void apply(Functor f) {
      //! \todo define a concept for "Updater"
      foreach(vertex_descriptor v, vertices()) potential(v).apply(f);
    }

    /*
     * Computes the entropy of the distribution (which base?)
     *
     * @return double representing the entropy of the distribution.
     *
     * \todo It would be really nice if we were able to compute
     *       conditional entropies and, perhaps, different base
     */
/*
    typename F::double entropy() const {
      double result = 0;
      foreach(vertex_descriptor v, vertices())
        result += marginal(v).entropy();
      foreach(edge_descriptor e, edges())
        result -= marginal(e).entropy();
      return result;
    }
*/

    /**
     * Computes the unnormalized log likelihood of the given assignment
     * according to this distribution.
     *
     * @param a    assignment to a subset of the variables of the distribution
     * @param base base of the log, default e
     * @return log likelihood of the assignment
     */
    double log_likelihood(const assignment& a, double base) const {
      using std::log;
      double loglike = 0;
      if (args.subset_of(a.keys())) {
        foreach(vertex_descriptor v, vertices())
          loglike += log(potential(v)(a));
      } else {
        foreach(vertex_descriptor v, vertices()) {
          domain d = potential(v).arguments().intersect(a.keys());
          loglike += log(potential(v).marginal(d)(a));
        }
      }
      return (loglike / log(base));
    }

    /**
     * Prints a human-readable representation of the cluster graph
     * model to the supplied output stream.
     *
     * \todo Standardize with factors.
     */
    template <typename OutputStream>
    void print(OutputStream& out) const { base::print(out); }

    /**
     * Checks that this cluster graph model is valid.  This method will
     * generate an assertion violation if the cluster graph model is
     * not valid.  The cluster graph must be valid.
     * \todo Should anything else be checked?
     */
    void check_validity() const {
      // Check that the cluster graph is valid.
      base::check_validity();
    }

    /**
     * Checks whether the cliques of this cluster graph model satisfy the
     * running intersection property.
     */
    void check_running_intersection() { base::check_validity(); }

    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

  }; // class cluster_graph_model

  /**
   * Prints a human-readable representation of the cluster graph model
   * to the supplied output stream.
   */
  template <typename Factor>
  std::ostream& operator<<(std::ostream& out,
                           const cluster_graph_model<Factor>& model) {
    model.print(out);
    return out;
  }

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CLUSTER_GRAPH_MODEL_HPP
