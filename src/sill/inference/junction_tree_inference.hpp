#ifndef SILL_JUNCTION_TREE_INFERENCE_HPP
#define SILL_JUNCTION_TREE_INFERENCE_HPP

#include <map>
#include <sill/factor/concepts.hpp>
#include <sill/factor/constant_factor.hpp>

#include <sill/graph/bidirectional.hpp>
#include <sill/graph/min_fill_strategy.hpp>
#include <sill/model/junction_tree.hpp>
#include <sill/model/interfaces.hpp>

#include <sill/range/transformed.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /** 
   * An engine that performs the multiplicative sum-product algorithm. 
   *
   * Warning: right now, normalization is performed using double,
   * rather than F::result_type.
   *
   * \ingroup inference
   */
  template <typename F>
  class shafer_shenoy {
    concept_assert((Factor<F>));

    // Public type declarations    
    //==========================================================================
  public:
    //! The type of variables in the factor's domain
    typedef typename F::variable_type variable_type;

    //! The factor's domain type
    typedef typename F::domain_type domain_type;

    //! The factor's assignment type
    typedef typename F::assignment_type assignment_type;

    //! The junction tree type used to store the cluster potentials and messages
    typedef junction_tree<variable_type*, F, bidirectional<F> > jt_type;

    //! The descriptors for the junction tree 
    typedef typename jt_type::vertex vertex;
    typedef typename jt_type::edge edge;

    // Private data members
    //==========================================================================
  private:
    //! The junction tree used to store the cluster potentials and messages
    jt_type jt;

    //! True if the inference has been performed
    bool calibrated;

    //! The class used to compute the messages
    struct message_functor {
      void operator()(edge e, jt_type& jt) {
        using std::endl;
        // Get the source and target vertices.
        vertex u = e.source();
        vertex v = e.target();

        F result = jt[u];
        
        foreach(edge in, jt.in_edges(u))
          if (in.source() != v) result *= jt[in].directed(in);

        jt[e].directed(e) = result.marginal(jt.separator(e));
      }
    };

    //! A class that normalizes the factor along a directed edge
    //! \todo We should use F::result_type here
    struct normalizer {
      double zinv;
      normalizer(double z) : zinv(1/z) {}
      void operator()(edge e, jt_type& jt) { 
        jt[e].directed(e) *= constant_factor(zinv);
      }
    };

    //! Initializes the clique potentials of the junction tree
    template <typename Range>
    void initialize_potentials(const Range& factors) {
      // Initialize the clique potentials to unity
      foreach(vertex v, jt.vertices())
        jt[v] = 1;

      // Multiply the factors of gm to cliques that cover them
      foreach(const typename Range::value_type& factor, factors) {
        vertex v = jt.find_clique_cover(factor.arguments());
        jt[v] *= factor;
      }
    }

    // Constructors
    //==========================================================================
  public:
    //! Constructs the Shafer-Shenoy engine for a given graphical model
    shafer_shenoy(const graphical_model<F>& gm) : calibrated(false) {
      // Create a junction tree with cliques that cover the factors of gm
      markov_graph<variable_type*> graph(gm.markov_graph());
      jt.initialize(graph, min_fill_strategy());

      // Initialize the potentials
      initialize_potentials(gm.factors());
    }

    //! Constructs the Shafer-Shenoy engine for a collection of factors
    //! \param factors Factors of a factorized model. The factors do not
    //!                need to be triangulated.
    //! Only the ranges with value_type and begin() members are supported
    template <typename Range>
    shafer_shenoy(const Range& factors, typename Range::value_type* = NULL)
      : calibrated(false) {
      concept_assert((ForwardRange<Range>));

      // Create a junction tree with cliques that cover the factors
      typedef typename Range::value_type factor_type;
      markov_graph<variable_type*> graph;
      foreach(const factor_type& f, factors) 
        graph.add_clique(f.arguments());
      jt.initialize(graph, min_fill_strategy());

      // Initialize the potentials
      initialize_potentials(factors);
    }

  #ifdef SWIG
    shafer_shenoy(const std::vector<F>& factors);
  #endif

    /** 
     * Constructs a Shafer-Shenoy engine for a given junction tree.
     */
    shafer_shenoy(const junction_tree<variable_type*, F>& factor_jt) 
      : calibrated(false) {
      foreach(vertex v, factor_jt.vertices()) {
        assert(factor_jt.clique(v).superset_of(factor_jt[v].arguments()));
        jt.add_clique(v, factor_jt.clique(v), factor_jt[v]);
      }
      foreach(edge e, factor_jt.edges()) {
        jt.add_edge(e.source(), e.target());
      }
    }

    // Queries
    //==========================================================================
    //! Returns the tree width of the underlying junction tree
    int tree_width() const {
      return jt.tree_width();
    }

    //! Performs the inference
    void calibrate() { 
      mpp_traversal(jt, message_functor());
      calibrated = true;
    }

    //! Normalizes all beliefs
    void normalize() {
      assert(calibrated);
      vertex root = jt.root();
      if (jt.num_vertices()>1) {
        // Compute the normalization constant z, and normalize the root
        // and every message in the direction from the root
        double z = belief(*jt.edges().first).norm_constant();
        assert(is_positive_finite(z));
        jt[root] *= constant_factor(1/z);
        pre_order_traversal(jt, root, normalizer(z));
      } else {
        // There is only one vertex, so all the factors are at the root
        jt[root].normalize();
      }
    }

    //! Conditions the inference on an assignment to one or more variables
    //! This is a mutable operation
    void condition(const assignment_type& a) {
      domain_type vars = a.keys();

      // Find all cliques that contain an old variable
      typename std::vector<vertex> vertices;
      jt.find_intersecting_cliques(vars, std::back_inserter(vertices));
      
      // Update each affected clique and separator
      foreach(vertex v, vertices) {
        jt[v] = jt[v].restrict(a);
        foreach(edge e, jt.out_edges(v)) {
          if (jt[e].forward.arguments().meets(vars)) 
            jt[e].forward = jt[e].forward.restrict(a);
          if (jt[e].reverse.arguments().meets(vars))
            jt[e].reverse = jt[e].reverse.restrict(a);
        }
        jt.set_clique(v, jt.clique(v) - vars);
      }
    }

    //! Returns the belief associated with a clique
    F belief(vertex v) const {
      assert(calibrated);
      F result = jt[v];
      foreach(edge in, jt.in_edges(v))
        result *= jt[in].directed(in);
      return result;
    }

    //! Returns the belief associated with a separator
    F belief(edge e) const {
      assert(calibrated);
      return jt[e].forward * jt[e].reverse;
    }

    /**
     * Returns the belief for a set of variables.
     * \throw std::invalid_argument 
     *        if the specified set is not covered by a clique of 
     *        the junction tree constructed by the engine.
     */
    F belief(const domain_type& vars) const {
      assert(calibrated);

      // Try to find a separator that covers vars
      edge e = jt.find_separator_cover(vars);
      if(e != edge()) return belief(e).marginal(vars);

      // Otherwise, look for a clique cover
      vertex v = jt.find_clique_cover(vars);
      assert(v);
      return belief(v).marginal(vars);
    }

    //! Returns the beliefs over the cliques
    std::vector<F> clique_beliefs() const {
      assert(calibrated);
      std::vector<F> result(jt.num_vertices()); 
      size_t i = 0;
      foreach(vertex v, jt.vertices())
        result[i++] = belief(v);
      return result;
    }

    //! Message along a directed edge
    const F& message(vertex u, vertex v) const {
      edge e = jt.get_edge(u, v);
      return jt[e].directed(e);
    }

    //! Message along a directed edge
    const F& message(edge e) const {
      return jt[e].directed(e);
    }

  };

  /**
   * Implements a junction tree inference engine with division algorithm. 
   * \ingroup inference
   */
  template <typename F>
  class hugin {
    concept_assert((Factor<F>));

    // Public type declarations    
    //==========================================================================
  public:
    //! The type of variables in the factor's domain
    typedef typename F::variable_type variable_type;

    //! The factor's domain type
    typedef typename F::domain_type domain_type;

    //! The type of junction tree that stores the potentials
    typedef junction_tree<variable_type*, F, F> jt_type;
    
    // Shortcuts
    typedef typename jt_type::vertex vertex;
    typedef typename jt_type::edge edge;
    
    // Private data members
    //==========================================================================
  private:
    jt_type jt;

    //! Distributes a collection of factors to cliques in the junction tree
    template <typename Range>
    void initialize_potentials(const Range& factors) {
      foreach(vertex v, jt.vertices())
        jt[v] = 1; // initialize vertex potentials to unity
      foreach(edge e, jt.edges())
        jt[e] = 1; // initialize the edge potentials to unity
      foreach(const F& factor, factors)
        jt[jt.find_clique_cover(factor.arguments())] *= factor;
    }

    /**
     * A functor which passes flows through the junction tree.
     * Given an edge e = (u, v), this function passes flow from u to v.
     */
    struct flow_functor {
      hugin* engine;
      flow_functor(hugin* engine) : engine(engine) {}
      void operator()(edge e, jt_type&) { engine->pass_flow(e); }
    };

    // Constructors
    //==========================================================================
  public:
    //! Constructs a ...
    /*
    hugin(const junction_tree<variable_type*>& jt_) : jt(jt_) {
      foreach(vertex v, jt.vertices())
        jt[v] = 1;
      foreach(edge e, jt.edges())
        jt[e] = 1;
    }
    */

    //! Constructs a Hugin engine for a graphical model
    hugin(const graphical_model<F>& gm) {
      markov_graph<variable_type*> mg = gm.markov_graph();
      jt.initialize(mg, min_fill_strategy());
      initialize_potentials(gm.factors());
    }

    //! Constructs a Hugin engine for a collection of factors
    hugin(const std::vector<F>& factors) {
      markov_graph<variable_type*> mg;
      foreach(const F& f, factors) mg.add_clique(f.arguments());
      jt.initialize(mg, min_fill_strategy());
      initialize_potentials(factors);
    }
    
    // Queries
    //==========================================================================
    //! Passes flow along a directed edge of the junction tree
    void pass_flow(edge e) {
      // Get the source and target vertices.
      vertex u = e.source();
      vertex v = e.target();
      
      // Compute the new separator potential using u and
      // update v's potential with ratio of the new and the old separator
      F new_sep_potential = jt[u].marginal(jt.separator(e));
      jt[v] *= new_sep_potential;
      jt[v] /= jt[e];
      jt[e]  = new_sep_potential;
      
      /* DEBUG */
      // Note: this would fail for Gaussian factors which may not be
      // initially normalizable; add a flag
      if (!jt[e].is_normalizable() ||
          !jt[v].is_normalizable()) {
        using std::endl;
        std::cerr << "Cannot normalize after flow.  Source potential:"
                  << endl << jt[u] << endl
                  << "separator potential: " << endl
                  << jt[e] << endl
                  << "target potential: " << endl
                  << jt[v] << endl;
        assert(false);
      }
    }

    //! Passes the flow from u to v
    void pass_flow(vertex u, vertex v) {
      pass_flow(jt.get_edge(u, v));
    }
 
    //! Calibrates the jt by passing flows using the message passing protocol.
    void calibrate() {
      mpp_traversal(jt, flow_functor(this));
    }

    //! Normalizes the clique and edge potentials
    void normalize() {
      foreach(vertex v, jt.vertices()) jt[v].normalize();
      foreach(edge e, jt.edges()) jt[e].normalize();
    } 

  #ifndef SWIG    
    //! Returns the junction tree
    const jt_type& tree() const {
      return jt;
    }
  #endif
    
    //! Mutable access to vertex potential
    //! The caller must not extend the arguments of the factor beyond the clique
    F& potential(vertex v) {
      return jt[v];
    }

    //! Returns the vertex potential
    const F& potential(vertex v) const {
      return jt[v];
    }
      
    //! Returns the clique potentials
    std::vector<F> clique_beliefs() const {
      std::vector<F> beliefs;
      beliefs.reserve(jt.num_vertices());
      foreach(vertex v, jt.vertices())
        beliefs.push_back(jt[v]);
      return beliefs;
    }

    // Modifierrs
    //==========================================================================
    /*
    void add_factor(const F& factor) { 
      jt[jt.find_clique_cover(factor.arguments())] *= factor;
    }
    */

  }; // class hugin

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
