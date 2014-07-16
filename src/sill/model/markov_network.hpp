#ifndef SILL_MARKOV_NETWORK_HPP
#define SILL_MARKOV_NETWORK_HPP

#include <iterator>
#include <set>
#include <map>

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/graph/property_functors.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/range/transformed.hpp>
#include <sill/range/joined.hpp>
#include <sill/serialization/list.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Implements a Markov network with pairwise potentials.
   * Note: the user must ensure that the arguments of edge and node factors
   * remain valid
   *
   * \param F
   *        The type of factors associated with the nodes of the markov network.
   *        Must satisfy the Factor concept.
   *
   * \ingroup model
   */
  template <typename F> 
  class pairwise_markov_network 
    : public markov_graph<typename F::variable_type*, F, F>,
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
    typedef typename F::assignment_type assignment_type;

    //! The base class
    typedef sill::markov_graph<variable_type*, F, F> base;

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

    //! Returns the factor associated with the node
    const F& factor(vertex v) const {
      return this->operator[](v);
    }

    //! Returns the factor associated with the node
    F& factor(vertex v) {
      return this->operator[](v);
    }
    
    //! Returns the factor associated with the edge
    const F& factor(edge e) const {
      return this->operator[](e);
    }

    //! Returns the factor associated with the edge
    F& factor(edge e) {
      return this->operator[](e);
    }

    //! Returns the factors associated with the nodes
    forward_range<const F&> node_factors() const {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    //! Returns the factors associated with the edges
    forward_range<const F&> edge_factors() const {
      return make_transformed(edges(), edge_property_functor(*this));
    }

    //! Returns the factors associated with this graphical model
    forward_range<const F&> factors() const {
      return make_joined
        (make_transformed(vertices(), vertex_property_functor(*this)),
         make_transformed(edges(), edge_property_functor(*this)));
    }
    
    // Queries
    //==========================================================================

    //! Evaluates the unnormalized log-likelihood of an assignment
    double log_likelihood(const assignment_type& a) const {
      using std::log;
      double result = 0;
      foreach(const F& factor, factors()) {
        result += factor.logv(a);
      }
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
      foreach(vertex v, vertices()) {
        assert(factor(v).arguments() == make_domain(v));
      }
      foreach(edge e, edges()) {
        assert(factor(e).arguments() == nodes(e));
      }
    }

    // Mutators
    //==========================================================================
    //! Extends the domains of all factors to include the incident nodes
    void extend_domains() {
      foreach(vertex v, vertices())
        if (!factor(v).arguments().count(v))
          factor(v) *= F(make_domain(v), 1);
      foreach(edge e, edges())
        if (!includes(factor(e).arguments(), nodes(e)))
          factor(e) *= F(nodes(e), 1);
    }

    //! Adds a factor to the graphical model and creates the vertices and edges
    void add_factor(const F& factor) {
      const domain_type& vars = factor.arguments();
      assert(vars.size() == 1 || vars.size() == 2);
      switch (vars.size()) {
        case 1:
	  this->add_vertex(*vars.begin(), factor);
          break;
        case 2:
	  this->add_edge(*vars.begin(), *++vars.begin(), factor);
          break;
        default:
          assert(false);
      }
    }

    //! Conditions the model on an assignment
    void condition(const assignment_type& a) {
      foreach(typename assignment_type::const_reference p, a) {
        variable_type* u = p.first;
        if (this->contains(u)) {
          foreach(edge e, this->out_edges(u)) {
            factor(e.target()) *= factor(e).restrict(a);
          }
          this->remove_vertex(u);
        }
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
    //! The type of the factor values
    typedef typename F::result_type result_type;

    //! The type of variables that form the factor's domain
    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! The assignment type of the factor
    typedef typename F::assignment_type assignment_type;

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

    bool operator==(const markov_network& other) {
      return base::operator==(other) && factors_ == other.factors_;
    }

    // Mutators
    //==========================================================================
    //! Saves the network to an archive
    void save(oarchive& ar) const {
      base::save(ar);
      ar << factors_;
    }
    
    //! Loads the graph from an archive
    void load(iarchive& ar) {
      base::load(ar);
      ar >> factors_;
    }
    
    void add_factor(const F& factor) {
      factors_.push_back(factor);
      base::add_clique(factor.arguments());
    }

    /**
     * Conditions this Markov network on an assignment to one or
     * more of its variables. This is a mutable operation.
     * Note this operation does not preserve the normalization constant.
     *
     * @param a
     *        An assignment to some variables.  This assignment is
     *        instantiated in each factor.
     *
     * TODO: unit test this function
     */
    markov_network& condition(const assignment_type& a) {
      // Compute the variables that are conditioned on.
      domain_type restricted_vars = set_intersect(keys(a), arguments());
      if (restricted_vars.empty()) {
        return *this;
      }

      // For each factor which includes a variable which is being restricted,
      // (handling factors which no longer have any arguments via const_f)
      typename std::list<F>::iterator it = factors_.begin();
      while (it != factors_.end()) {
        if (set_disjoint(restricted_vars, it->arguments())) {
          ++it;
        } else if (includes(restricted_vars, it->arguments())) {
          factors_.erase(it++);
        } else {
          *it = it->restrict(a);
          ++it;
        }
      }

      // Remove the nodes for the restricted variables.
      foreach(variable_type* v, restricted_vars) {
        this->remove_node(v);
      }

      return *this;
    }

  }; // class markov_network
}

#include <sill/macros_undef.hpp>

#endif
