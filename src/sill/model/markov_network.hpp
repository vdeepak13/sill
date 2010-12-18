#ifndef SILL_MARKOV_NETWORK_HPP
#define SILL_MARKOV_NETWORK_HPP

#include <iterator>
#include <set>
#include <map>

#include <boost/serialization/list.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/constant_factor.hpp>
#include <sill/graph/property_functors.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/range/transformed.hpp>
#include <sill/range/joined.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Implements a Markov network with pairwise potentials.
   * Note: the user must ensure that the arguments of edge and node factors
   * remain valid
   *
   * \param NodeF 
   *        The type of factors associated with the nodes of the markov network.
   *        Must satisfy the Factor concept.
   * \param EdgeF
   *        The type of factors associated with the edges of the markov network.
   *        Must satisfy the Factor concept.
   *
   * \ingroup model
   */
  template <typename NodeF, typename EdgeF = NodeF> 
  class pairwise_markov_network 
    : public markov_graph<typename NodeF::variable_type*, NodeF, EdgeF>,
      public graphical_model<NodeF> {

    concept_assert((Factor<NodeF>));
    concept_assert((Factor<EdgeF>));
    static_assert((boost::is_same<typename NodeF::variable_type,
                                  typename EdgeF::variable_type>::value));

    // Public type declarations
    //==========================================================================
  public:
    //! The type of variables that form the factor's domain
    typedef typename NodeF::variable_type variable_type;

    //! The domain type of the factor
    typedef typename NodeF::domain_type domain_type;

    //! The assignment type of the factor
    typedef std::map<variable_type*, typename variable_type::value_type> 
      assignment_type;

    //! The base class
    typedef sill::markov_graph<variable_type*, NodeF, EdgeF> base;

    // Shortcuts
    typedef typename base::vertex vertex;
    typedef typename base::edge edge;
    using base::nodes;
    using base::vertices;
    using base::edges;

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Default constructor
    pairwise_markov_network() { }

    pairwise_markov_network(const domain_type& variables)
      : base(variables) { }

    //! Constructs a Markov network with the given variables and edges
    //! Each edge is specified as std::pair<vertex, vertex>
    template <typename EdgeRange>
    pairwise_markov_network(const domain_type& variables,
                            const EdgeRange& edges)
      : base(variables, edges) { }

    /*
    #ifndeg SWIG
    //! Constructs a Markov network with the given graph structure
    template <typename VP, typename EP, typename Tag>
    pairwise_markov_network(const markov_graph<VP, EP, Tag>& g) : base(g) { }
    #endif
    */
    /*
    //! Constructs a Markov network with the given graph structure
    template <typename Graph, typename VariableMap>
    pairwise_markov_network(const Graph& g, VariableMap m) : base(g, m) { }
    */

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    // Accessors
    //==========================================================================
    //! Returns the arguments of the model
    domain_type arguments() const {
      return base::nodes();
    }

    //! Returns the factor associated with a vertex or edge
    const NodeF& factor(const vertex v) const {
      return base::operator[](v);
    }
    const EdgeF& factor(const edge e) const {
      return base::operator[](e);
    }

    NodeF& factor(const vertex v) {
      return base::operator[](v);
    }

    EdgeF& factor(const edge e) {
      return base::operator[](e);
    }

    forward_range<const NodeF&> node_factors() const {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    forward_range<const EdgeF&> edge_factors() const {
      return make_transformed(edges(), edge_property_functor(*this));
    }

    // GraphicalModel interface
    forward_range<NodeF&> node_factors() {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    forward_range<EdgeF&> edge_factors() {
      return make_transformed(edges(), edge_property_functor(*this));
    }

    //! Returns the factors associated with this graphical model
    //! This function is only supported if NodeF is the same type as EdgeF
    forward_range<const NodeF&> factors() const {
      if (boost::is_same<NodeF, EdgeF>::value) {
        return make_joined
          (make_transformed(vertices(), vertex_property_functor(*this)),
           make_transformed(edges(), edge_property_functor(*this)));
      } else {
        assert(false); // unsupported
      }
    }
    
    // Queries
    //==========================================================================

    //! Evaluates the unnormalized log-likelihood of an assignment
    double log_likelihood(const assignment_type& a) const {
      using std::log;
      double result = 0;
      foreach(vertex v, vertices()) 
        result += factor(v).logv(a);
      foreach(edge e, edges())
        result += factor(e).logv(a);
      return result;
    }

    //! Evaluates the unnormalized likelihood of an assignment
    logarithmic<double> operator()(const assignment_type& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    bool d_separated(const domain_type& x, const domain_type& y,
                     const domain_type& z = domain_type::empty_set) const {
      return base::d_separated(x, y, z);
    }

    sill::markov_graph<variable_type*> markov_graph() const {
      return sill::markov_graph<variable_type*>(*this);
    }

    //! Throws an assertion violation if the MRF is not valid
    void check() const {
      foreach(vertex v, vertices())
        assert( factor(v).arguments() == node(v) );
      foreach(edge e, edges()) {
        domain_type args = make_domain(source_node(e), target_node(e));
        assert( factor(e).arguments() == args );
      }
    }

    // Mutators
    //==========================================================================
    //! Extends the domains of all factors to include the incident nodes
    void extend_domains() {
      foreach(vertex v, vertices())
        if (!factor(v).arguments().count(v))
          factor(v) *= NodeF(make_domain(v), 1);
      foreach(edge e, edges())
        if (!includes(factor(e).arguments(), nodes(e)))
          factor(e) *= EdgeF(nodes(e), 1);
    }

    //! Adds a factor to the graphical model and creates the vertices and edges
    template<typename Factor>
    void add_factor(const Factor& f) {
      domain_type vars = f.arguments();
      assert(vars.size() == 1 || vars.size() == 2);
      this->add_clique(vars);
      switch (vars.size()) {
        case 1:
	  base::add_vertex((*vars.begin()), f);
          break;
        case 2:
	  base::add_edge((*vars.begin()), (*boost::next(vars.begin(), 1)), f);
          break;
        default:
          assert(false);
      }
    }

  }; // class pairwise_markov_network

  /**
   * Implements a Markov network over arbitrary cliques.
   *
   * \ingroup model
   */
  template <typename F>
  class markov_network
    : public markov_graph<typename F::variable_type*>, 
      public graphical_model<F> {
    concept_assert((Factor<F>));

    // Public type declarations
    //==========================================================================
  public:
    //! The type of variables that form the factor's domain
    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! The assignment type of the factor
    typedef std::map<variable_type*, typename variable_type::value_type> 
      assignment_type;

    //! The base class
    typedef sill::markov_graph<variable_type*> base;

    // Shortcuts
    typedef typename base::vertex vertex;
    typedef typename base::edge edge;

    // Private data members
    //==========================================================================
  private:
    //! The factors that comprise this network
    std::list<F> factors_;

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Creates an empty Markov network
    markov_network() { }

    //! Creates an empty Markov network with given variables
    markov_network(const domain_type& variables) : base(variables) { }

    //! Creates a Markov network with given potentials.
    template <typename FactorRange>
    markov_network(const FactorRange& factors)
      : factors_(boost::begin(factors), boost::end(factors)) {
      foreach(const F& factor, factors_)
        base::add_clique(factor.arguments());
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      assert(false); // TODO
      return std::string();
      // std::ostringstream out; out << *this; return out.str(); 
    }

    // Accessors
    //==========================================================================
    domain_type arguments() const {
      return base::nodes();
    }

    forward_range<const F&> factors() const {
      return factors_;
    }

    // Queries
    //==========================================================================
    double log_likelihood(const assignment_type& a) const {
      using std::log;
      double result = 0;
      foreach(const F& factor, factors_) result += factor.logv(a);
      return result;
    }

    logarithmic<double> operator()(const assignment_type& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    sill::markov_graph<variable_type*> markov_graph() const {
      return sill::markov_graph<variable_type*>(*static_cast<const base*>(this));
    }

    bool d_separated(const domain_type& x, const domain_type& y,
                     const domain_type& z = domain_type::empty_set) const {
      return base::d_separated(x, y, z);
    }

    //! Prints the graph structure, and optionally the factors as well.
    template <typename OutputStream>
    void print(OutputStream& out, bool print_factors = false) const {
      out << *this;
      if (print_factors) {
        foreach(F f, factors()) out << f;
      }
    }

    // Mutators
    //==========================================================================
    void add_factor(const F& factor) {
      factors_.push_back(factor);
      base::add_clique(factor.arguments());
    }

    /**
     * Conditions this Markov net on an assignment to one or
     * more of its variables. This is a mutable operation.
     * Note this does not normalize the distribution.
     *
     * @param a
     *        An assignment to some variables.  This assignment is
     *        instantiated in each factor.
     * \todo Should we combine factors which now have the same argument sets?
     */
    markov_network& condition(const assignment_type& a) {

      // Compute the variables that are conditioned on.
      domain_type restricted_vars = set_intersect(keys(a), arguments());
      if (restricted_vars.empty())
        return *this;

      // For each factor which includes a variable which is being restricted,
      // (handling factors which no longer have any arguments via const_f)
      double const_f = 1;
      std::list<F> new_factors;
      foreach(F& f, factors_) {
        if (set_disjoint(f.arguments(), restricted_vars))
          new_factors.push_back(f);
        else {
          if (includes(keys(a), f.arguments()))
            const_f *= f.v(a);
          else
            new_factors.push_back(f.restrict(a));
        }
      }
      if (new_factors.empty())
        new_factors.push_back(F(const_f));
      else
        new_factors.front() *= constant_factor(const_f);
      factors_.swap(new_factors);

      // Remove the nodes for the restricted variables.
      foreach(variable_type* v, restricted_vars)
        remove_node(v);

      return *this;
    }

  }; // class markov_network
}

#include <sill/macros_undef.hpp>

#endif
