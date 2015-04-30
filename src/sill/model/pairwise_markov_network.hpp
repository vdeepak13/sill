#ifndef SILL_PAIRWISE_MARKOV_NETWORK_HPP
#define SILL_PAIRWISE_MARKOV_NETWORK_HPP

#include <sill/global.hpp>
#include <sill/graph/property_functors.hpp>
#include <sill/graph/undirected_graph.hpp>
#include <sill/math/logarithmic.hpp>
#include <sill/range/transformed.hpp>
#include <sill/traits/is_range.hpp>

namespace sill {

  /**
   * Implements a Markov network with pairwise potentials.
   *
   * \tparam F The type of factors associated with the vetices and edges.
   *           Must model the Factor concept.
   *
   * \ingroup model
   */
  template <typename F> 
  class pairwise_markov_network 
    : public undirected_graph<typename F::variable_type*, F, F> {

    typedef undirected_graph<typename F::variable_type*, F, F> base;
  
    // Public type declarations
    //==========================================================================
  public:
    // FactorizedModel types
    typedef typename F::real_type       real_type;
    typedef logarithmic<real_type>      result_type;
    typedef typename F::variable_type   variable_type;
    typedef typename F::domain_type     domain_type;
    typedef typename F::assignment_type assignment_type;
    typedef F                           value_type;

    // Shortcuts
    typedef typename base::vertex_type vertex_type;
    typedef typename base::edge_type edge_type;

    // Constructors
    //==========================================================================
  public:
    //! Default constructor. Creates an empty pairwise Markov network.
    pairwise_markov_network() { }

    //! Constructs a pairwise Markov network with given variables and no edges.
    explicit pairwise_markov_network(const domain_type& variables) {
      for (variable_type* v : variables) {
        this->add_vertex(v);
      }
    }

    //! Constructs a pairwise Markov network from a collection of factors.
    template <typename Range>
    explicit pairwise_markov_network(
      const Range& factors,
      typename std::enable_if<is_range<Range, F>::value>::type* = 0) {
      for (const F& factor : factors) {
        add_factor(factor);
      }
    }

    // Accessors
    //==========================================================================

    //! Returns the arguments of this model (i.e., the range of all vertices).
    iterator_range<typename base::vertex_iterator> arguments() const {
      return this->vertices();
    }

    //! Returns the arguments of the factor associated with a vertex.
    const domain_type& arguments(variable_type* v) const {
      return (*this)[v].arguments();
    }

    //! Returns the arguments of the factor associated with an edge.
    const domain_type& arguments(const edge_type& e) const {
      return (*this)[e].arguments();
    }
    
    //! Returns the factors associated with vertices.
    iterator_range<
      boost::transform_iterator<
        vertex_property_fn<const base>, typename base::vertex_iterator> >
    node_factors() const {
      return make_transformed(this->vertices(),
                              vertex_property_fn<const base>(this));
    }

    //! Returns the factors associated with the edges.
    iterator_range<
      boost::transform_iterator<
        edge_property_fn<const base>, typename base::edge_iterator> >
    edge_factors() const {
      return make_transformed(this->edges(),
                              edge_property_fn<const base>(this));
    }

    // Queries
    //==========================================================================

    /**
     * Returns the unnormalized likelihood of the given assignment.
     * The assignment must include all the arguments of this Markov network.
     */
    result_type operator()(const assignment_type& a) const {
      return result_type(log(a), log_tag());
    }

    /**
     * Returns the unnormalized log-likelihood of the given assignment.
     * The assignment must include all the arguments of this Markov network.
     */
    real_type log(const assignment_type& a) const {
      real_type result(0);
      for (const F& f : node_factors()) {
        result += f.log(a);
      }
      for (const F& f : edge_factors()) {
        result += f.log(a);
      }
      return result;
    }

    /**
     * Computes a minimal Markov graph capturing dependencies in this model.
     */
    void markov_graph(undirected_graph<variable_type*>& mg) const {
      for (vertex_type v : this->vertices()) {
        mg.add_vertex(v);
      }
      for (edge_type e : this->edges()) {
        mg.add_edge(e.source(), e.target());
      }
    }

    /**
     * Returns true if the domain of the factor matches the vertex / vertices.
     */
    bool valid() const {
      for (vertex_type v : this->vertices()) {
        if (!equivalent(arguments(v), domain_type({v}))) {
          return false;
        }
      }
      for (edge_type e : this->edges()) {
        if (!equivalent(arguments(e), domain_type({e.source(), e.target()}))) {
          return false;
        }
      }
      return true;
    }

    // Mutators
    //==========================================================================
    /**
     * Initializes the node and edge potentials with the given functor.
     * This is performed by invoking the functor on the arguments given
     * by each vertex and edge and assigning the result to the model.
     */
    void initialize(std::function<F(const domain_type&)> fn) {
      for (vertex_type v : this->vertices()) {
        (*this)[v] = fn({v});
      }
      for (edge_type e : this->edges()) {
        (*this)[e] = fn({e.source(), e.target()});
      }
    }

    /**
     * Extends the domains of all factors to include the incident vertices.
     */
    void extend_domains() {
      typename F::result_type one(1);
      for (vertex_type v : this->vertices()) {
        if (!arguments(v).count(v)) {
          (*this)[v] = F({v}, one) * (*this)[v];
        }
      }
      for (edge_type e : this->edges()) {
        if (!arguments(e).count(e.source()) ||
            !arguments(e).count(e.target())) {
          (*this)[e] = F({e.source(), e.target()}, one) * (*this)[e];
        }
      }
    }

    /**
     * Adds a factor to the graphical model and creates the vertices and edges.
     * If the corresponding vertex/edge already exists, does nothing.
     *
     * \return bool indicating whether the factor was inserted
     * \throw std::invalid_argument if the factor is not unary or binary
     */
    bool add_factor(const F& factor) {
      const domain_type& args = factor.arguments();
      switch (args.size()) {
        case 1:
	  return this->add_vertex(*args.begin(), factor);
        case 2:
	  return this->add_edge(*args.begin(), *++args.begin(), factor).second;
        default:
          throw std::invalid_argument("Unsupported factor arity " +
                                      std::to_string(args.size()));
      }
    }

    /**
     * Conditions the model on an assignment. This restricts any edge
     * factor whose argument is contained in a and multiplies it to
     * the adjacent node factor. The normalizing constant is not preserved.
     */
    void condition(const assignment_type& a) {
      for (const auto& p : a) {
        variable_type* u = p.first;
        if (this->contains(u)) {
          for (edge_type e : this->out_edges(u)) {
            if (!a.count(e.target())) {
              (*this)[e.target()] *= (*this)[e].restrict(a);
            }
          }
          this->remove_vertex(u);
        }
      }
    }
    
  }; // class pairwise_markov_network

} // namespace sill

#endif
