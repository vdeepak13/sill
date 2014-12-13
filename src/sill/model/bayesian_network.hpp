#ifndef SILL_BAYESIAN_NETWORK_HPP
#define SILL_BAYESIAN_NETWORK_HPP
#include <map>

#include <boost/random/uniform_real.hpp>

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/learning/dataset_old/dataset.hpp>
#include <sill/model/bayesian_graph.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/model_functors.hpp>
#include <sill/graph/algorithm/graph_traversal.hpp>
#include <sill/graph/property_functors.hpp>

#include <sill/range/transformed.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A Bayesian network with CPDs for each variable.
   * 
   * Note: The user must ensure that the arguments of edge and node factors
   * remain valid.
   *
   * \ingroup model
   */
  template <typename F>
  class bayesian_network : 
    public bayesian_graph<typename F::variable_type*, F>,
    public graphical_model<F>  {

    concept_assert((Factor<F>));

    // Public type declarations
    // =========================================================================
  public:

    //! The types of variables, etc. used by the factor.
    typedef typename F::variable_type    variable_type;
    typedef typename F::domain_type      domain_type;
    typedef typename F::assignment_type  assignment_type;
    typedef typename F::record_type      record_type;

    //! The primary base class
    typedef bayesian_graph<variable_type*, F> base;

    // Shortcuts
    typedef typename base::edge edge;
    typedef typename base::vertex vertex;
    using base::vertices;

    // Constructors
    // =========================================================================
  public:

    //! Default constructor; creates an empty Bayes net.
    bayesian_network() { }

    /**
     * Constructor:
     * Creates a Bayes net with the given set of nodes and no edges.
     * (The factors must be added later.)
     */
    bayesian_network(const domain_type& variables) : base(variables) { }

    //! Constructs a Bayes net with the given graph structure.
    bayesian_network(const bayesian_graph<variable_type*>& g) : base(g) { }

    operator std::string() const {
      assert(false);
      //std::ostringstream out; out << *this; return out.str(); 
      return std::string();
    }

    // Queries
    // =========================================================================

    domain_type arguments() const {
      return base::nodes();
    }

    //! Returns the factor associated with a variable
    const F& factor(vertex v) const {
      return this->operator[](v);
    }

    //! Returns the factor associated with a variable
    F& factor(vertex v) {
      return this->operator[](v);
    }

    forward_range<F&> factors() {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    forward_range<const F&> factors() const {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    bool d_separated(const domain_type& x, const domain_type& y,
                     const domain_type& z = domain_type::empty_set) const {
      return base::d_separated(x, y, z);
    }

    sill::markov_graph<variable_type*> markov_graph() const {
      sill::markov_graph<variable_type*> mg;
      foreach(vertex v, vertices())
        mg.add_clique(factor(v).arguments());
      return mg;
    }

    /**
     * Throws an assertion violation if the following do not hold:
     * 1) The arguments of the factors correspond to vertices.
     * 2) The CPTs represent valid conditional probability distributions.
     */
    void check() const {
      foreach(vertex v, vertices()) {
        domain_type parents(boost::begin(this->parents(v)),
                            boost::end(this->parents(v)));
        assert(factor(v).arguments() == set_union(parents, v));
        assert(factor(v).is_conditional(parents));
      }
    }

    // Probabilistic model queries
    //==========================================================================

    //! Log likelihood (using the given base)
    double log_likelihood(const assignment_type& a, double base) const {
      assert(base > 0);
      double ll = 0;
      foreach(const F& f, factors())
        ll += f.logv(a);
      return ll / std::log(base);
    }

    //! Log (base e) likelihood
    double log_likelihood(const assignment_type& a) const {
      return log_likelihood(a, std::exp(1.0));
    }

    //! Log likelihood (using the given base)
    double log_likelihood(const record_type& r, double base) const {
      assert(base > 0);
      double ll = 0;
      foreach(const F& f, factors())
        ll += f.logv(r);
      return ll / std::log(base);
    }

    //! Log (base e) likelihood
    double log_likelihood(const record_type& r) const {
      return log_likelihood(r, std::exp(1.0));
    }

    logarithmic<double> operator()(const assignment_type& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    logarithmic<double> operator()(const record_type& r) const {
      return logarithmic<double>(log_likelihood(r), log_tag());
    }

    //! Returns a sample from a Bayes net over discrete variables.
    template <typename RandomNumberGenerator>
    assignment_type sample(RandomNumberGenerator& rng) const {
      assignment_type a;
      foreach(vertex v, directed_partial_vertex_order(*this)) {
        F f(factor(v));
        f = f.restrict(a);
        f.normalize();
        assignment_type tmpa(f.sample(rng));
        a[v] = tmpa[v];
      }
      return a;
    }

    // Modifiers
    // =========================================================================

    /**
     * Adds a factor P(v | other variables) to the graphical model
     * and creates the necessary vertices and edges.
     * If another factor P(v | some vars) exists in the model, then
     * this factor is multiplied with the current factor to create a new
     * factor P(v | other variables union some vars).
     * 
     * Note: It is the responsibility of the caller to ensure that the
     * graph remains a DAG.
     */
    void add_factor(variable_type* v, const F& f) {
      domain_type vars = f.arguments();
      assert(vars.count(v));
      vars = set_difference(vars, make_domain(v));
      base::add_family(v, vars);
      if (factor(vertex(v)).arguments().empty())
        factor(vertex(v)) = f;
      else
        factor(vertex(v)) *= f;
    }

    /**
     * Condition this model on the values in the given assignment
     * for the variables in restrict_vars.
     * This renormalizes the factors.
     *
     * @todo Make this more efficient.
     *
     * @bug Gaussian and table factors do not handle normalization in the same
     *      way since table factors do not have the concept of head/tail vars.
     *      Fix that!
     */
    bayesian_network&
    condition(const assignment_type& a_, const domain_type& restrict_vars) {
      assignment_type a;
      foreach(variable_type* v, restrict_vars)
        a[v] = safe_get(a_, v);

      bayesian_network tmp_bn;
      foreach(variable_type* v, vertices()) {
        if (includes(restrict_vars, factor(v).arguments())) {
          // Do nothing.
        } else if (restrict_vars.count(v)) {
          // Some parents of v remain.
          assert(false); // Not yet implemented.
        } else {
          // v and maybe some parents remain.
          F tmpf(factor(v).restrict(a));
          // BUG: See bug comment above in the comments for this method!
          tmp_bn.add_factor(v, tmpf);
        }
      }
      *this = tmp_bn;
      return *this;
    } // condition(a, restrict_vars)

    /**
     * Computes the conditional log likelihood: log P(y|x),
     * where this model represents P(Y,X).
     *
     * @param X    Variables (which MUST be a subset of this model's arguments)
     *             to condition on.
     * @param base base of the log (default = e)
     */
    double conditional_log_likelihood(const record_type& r,
                                      const domain_type& X,
                                      double base = exp(1.)) const {
      domain_type YX(arguments());
      assert(includes(r.variables(), YX));
      assert(includes(YX, X));
      bayesian_network tmpbn(*this);
      tmpbn.condition(r, X);
      return tmpbn.log_likelihood(r, base);
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected conditional log likelihood E[log P(Y|X)].
     */
    model_conditional_log_likelihood_functor<bayesian_network>
    conditional_log_likelihood(const domain_type& X,
                               double base = exp(1.)) const {
      return
        model_conditional_log_likelihood_functor<bayesian_network>
        (*this, X, base);
    }

  }; // class bayesian_network

  /**
   * Create a Markov network from a Bayes net.
   * \todo Make this a conversion constructor
   */
  template <typename F>
  markov_network<F>
  bayes2markov_network(const bayesian_network<F>& bn) {
    markov_network<F> mn;
    foreach(typename F::variable_type* v, bn.vertices())
      mn.add_factor(bn.factor(v));
    return mn;
  }

} // namespace sill

namespace boost {

  //! A traits class that lets bayesian_network work in BGL algorithms
  template <typename F>
  struct graph_traits< sill::bayesian_network<F> >
    : public graph_traits< sill::bayesian_graph<typename F::variable_type*, F> >
  { };

} // namespace boost


#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_BAYESIAN_NETWORK_HPP
