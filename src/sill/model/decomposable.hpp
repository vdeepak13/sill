#ifndef SILL_DECOMPOSABLE_HPP
#define SILL_DECOMPOSABLE_HPP

#include <vector>
#include <map>

#include <boost/range/iterator_range.hpp> // this will be removed

#include <sill/range/concepts.hpp>
#include <sill/factor/util/arguments_functor.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/algorithm/min_degree_strategy.hpp>
#include <sill/graph/algorithm/tree_traversal.hpp>
#include <sill/learning/dataset_old/dataset.hpp>
#include <sill/learning/evaluation/error_measures.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/model/junction_tree.hpp>
#include <sill/model/model_functors.hpp>
#include <sill/model/normalization_error.hpp>
#include <sill/inference/exact/variable_elimination.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // #define SILL_VERBOSE // for debugging

  // Forward declaration
  class table_factor;

  namespace impl {

    /**
     * Struct specifying which factors should be normalized more often.
     * This struct's value is false by default.
     * If this struct's value is true, then factors are normalized at these
     * additional times:
     *  - whenever factors are added,
     *  - during the post-traversal of calibration.
     */
    template <typename F>
    struct decomposable_extra_normalization {
      static const bool value = false;
    };

    template <>
    struct decomposable_extra_normalization<table_factor> {
      static const bool value = true;
    };

  } // namespace impl

  /**
   * A decomposable representation of a probability distribution.
   * Conceptually, decomposable model is a junction tree, in which
   * each vertex and each edge is associated with a factor of type F.
   *
   * Note: to ensure the stability of descriptors and iterators
   *       after copying, use the supplied custom copy constructor.
   *
   * @tparam F a type that models the DistributionFactor concept
   *
   * \todo Do we want automatic renormalization? Maybe could be
   *       a parameter to the decomposable model.
   *
   * \ingroup model
   */
  template <typename F>
  class decomposable : public graphical_model<F>
  {
    concept_assert((DistributionFactor<F>));
    template <typename T> friend class decomposable_iterator;

    // Public type declarations
    // =========================================================================
  public:
    //! The type of factors used in this decomposable model
    typedef F factor_type;

    //! The type of variables associated with a factor
    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! The assignment type of the factor
    typedef typename graphical_model<F>::assignment_type assignment_type;

    //! The record type of the factor
    typedef typename graphical_model<F>::record_type record_type;

    //! The underlying graph type
    typedef junction_tree<variable_type*, F, F> jt_type;

    // Graph types (copied from junction_tree)
    typedef size_t                  vertex;
    typedef undirected_edge<size_t> edge;
    typedef F                       vertex_property;
    typedef F                       edge_property;

    // Graph iterators
    typedef typename jt_type::vertex_iterator    vertex_iterator;
    typedef typename jt_type::neighbor_iterator  neighbor_iterator;
    typedef typename jt_type::edge_iterator      edge_iterator;
    typedef typename jt_type::in_edge_iterator   in_edge_iterator;
    typedef typename jt_type::out_edge_iterator  out_edge_iterator;

    // Protected type declarations and data members
    // =========================================================================
  protected:

    //! The argument variables of this distribution.
    domain_type args;

    //! The underlying junction tree
    junction_tree<variable_type*, F, F> jt;

    // Constructors and basic member functions
    // =========================================================================
  public:

    /**
     * Default constructor. The distribution has no arguments and
     * is identically one.
     */
    decomposable() { }

    /**
     * Initializes the decomposable model to the given set of clique marginals.
     * The marginals must be triangulated.
     */
    template <typename FactorRange>
    explicit decomposable(const FactorRange& factors,
                          typename FactorRange::iterator* = 0) {
      initialize(factors);
    }

    //! Virtual destructor to support approx_decomposable (experimental).
    virtual ~decomposable() { }

    /**
     * Initializes the decomposable model to the given set of clique marginals
     * and normalizes the model.
     * The marginals must be triangulated.
     */
    template <typename FactorRange>
    void initialize(const FactorRange& factors) {
      concept_assert((InputRangeConvertible<FactorRange, F>));

      // Initialize the clique marginals and the tree structure
      jt.initialize(make_transformed(factors, arguments_functor<F>()),
                    boost::begin(factors));

      // Depending on the factor type, pre-normalize clique marginals
      // to avoid numerical issues.
      if (impl::decomposable_extra_normalization<F>::value) {
        foreach(const vertex& v, vertices()) {
          if (jt[v].is_normalizable())
            jt[v].normalize();
        }
      }

      // Compute the separator marginals
      foreach(edge e, edges()) {
        vertex s = e.source(), t = e.target();
        if (clique(s).size() < clique(t).size())
          jt[e] = potential(s).marginal(separator(e));
        else
          jt[e] = potential(t).marginal(separator(e));
      }

      // Compute the arguments
      args.clear();
      foreach(vertex v, vertices()) {
        args.insert(clique(v).begin(), clique(v).end());
      }

      calibrate(); // Do this since the factors might not be marginals.
      normalize();
    }

    /**
     * Initializes the decomposable model to the given structure and
     * the given set of marginals and normalizes the model.
     * The length and the order of the marginals must exactly match
     * the vertices() range of the specified junction tree.
     *
     * @param structure  Only the nodes (structure) of this tree are used;
     *                   the vertex and edge properties are ignored.
     */
    template <typename VertProp, typename EdgeProp>
    void initialize
    (const junction_tree<variable_type*, VertProp, EdgeProp>& structure,
     const std::vector<F>& marginals) {

      jt.initialize(structure);
      args.clear();
      assert(marginals.size() == jt.num_vertices());
      size_t i = 0;
      foreach(vertex v, structure.vertices()) {
        const domain_type& argsi = marginals[i].arguments();
        jt[v] = marginals[i];
        args.insert(argsi.begin(), argsi.end());
        assert(argsi == jt.clique(v));
        i++;
      }

      // Depending on the factor type, pre-normalize clique marginals
      // to avoid numerical issues.
      if (impl::decomposable_extra_normalization<F>::value) {
        foreach(const vertex& v, vertices()) {
          if (jt[v].is_normalizable())
            jt[v].normalize();
        }
      }

      foreach(edge e, jt.edges())
        jt[e] = jt[e.source()].marginal(jt.separator(e));

      calibrate();
      normalize();
    }

    //! Swaps two decomposable models.
    void swap(decomposable& other) {
      jt.swap(other.jt);
      args.swap(other.args);
    }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << args << jt;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> args >> jt;
    }

    bool operator==(const decomposable& other) const {
      return args == other.args && jt == other.jt;
    }

    bool operator!=(const decomposable& other) const {
      return !operator==(other);
    }

    // Graph accessors
    // =========================================================================
    //! Returns an ordered set of all vertices
    std::pair<vertex_iterator, vertex_iterator>
    vertices() const {
      return jt.vertices();
    }

    //! Returns the vertices adjacent to u
    std::pair<neighbor_iterator, neighbor_iterator>
    neighbors(const vertex& u) const {
      return jt.neighbors(u);
    }

    //! Returns all edges in the graph
    std::pair<edge_iterator, edge_iterator>
    edges() const {
      return jt.edges();
    }

    //! Returns the edges incoming to a vertex, such that e.target() == u.
    std::pair<in_edge_iterator, in_edge_iterator>
    in_edges(const vertex& u) const {
      return jt.in_edges(u);
    }

    //! Returns the outgoing edges from a vertex, such that e.source() == u.
    std::pair<out_edge_iterator, out_edge_iterator>
    out_edges(const vertex& u) const {
      return jt.out_edges(u);
    }

    //! Returns true if the graph contains the given vertex
    bool contains(const vertex& u) const {
      return jt.contains(u);
    }

    //! Returns true if the graph contains an undirected edge {u, v}
    bool contains(const vertex& u, const vertex& v) const {
      return jt.contains(u, v);
    }

    //! Returns true if the graph contains an undirected edge
    bool contains(const edge& e) const {
      return jt.contains(e);
    }

    //! Returns an undirected edge with e.source()==u and e.target()==v.
    //! The edge must exist.
    edge get_edge(const vertex& u,  const vertex& v) const {
      return jt.get_edge(u, v);
    }

    //! Returns the number of edges adjacent to a vertex
    size_t in_degree(const vertex& u) const {
      return jt.in_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex
    size_t out_degree(const vertex& u) const {
      return jt.out_degree(u);
    }

    //! Returns the number of edges adjacent to a vertex
    size_t degree(const vertex& u) const {
      return jt.degree(u);
    }

    //! Returns true if the graph has no vertices
    bool empty() const {
      return jt.empty();
    }

    //! Returns the number of vertices
    size_t num_vertices() const {
      return jt.num_vertices();
    }

    //! Returns the number of edges
    size_t num_edges() const {
      return jt.num_edges();
    }

    //! Given an undirected edge (u, v), returns the equivalent edge (v, u)
    edge reverse(const edge& e) const {
      return jt.reverse(e);
    }

    //! Returns the marginal associated with a vertex
    const F& operator[](const vertex& u) const {
      return jt[u];
    }

    //! Returns the marginal associated with an edge
    const F& operator[](const edge& e) const {
      return jt[e];
    }

    //! Returns a null vertex
    static vertex null_vertex() { return jt_type::null_vertex(); }

    //! Returns the underlying junction tree.
    const jt_type& get_junction_tree() const { return jt; }

    // Junction tree queries
    //==========================================================================

    //! Returns the clique associated with a vertex.
    const domain_type& clique(vertex v) const {
      return jt.clique(v);
    }

    //! Returns the separator associated with an edge.
    const domain_type& separator(edge e) const {
      return jt.separator(e);
    }

    //! Returns the collection of the cliques
    forward_range<const domain_type&> cliques() const {
      return jt.cliques();
    }

    //! Returns the collection of the separators
    forward_range<const domain_type&> separators() const {
      return jt.separators();
    }

    //! Returns the maximum clique size minus one.
    int tree_width() const {
      return jt.tree_width();
    }

    //! Returns a vertex whose clique is a subset of the supplied
    //! set of variables.  \see junction_tree::find_clique_cover
    vertex find_clique_cover(const domain_type& set) const {
      return jt.find_clique_cover(set);
    }

    //! Returns a vertex whose clique intersects the supplied set of variables.
    //! \see junction_tree::find_clique_meets
    vertex find_clique_meets(const domain_type& set) const {
      return jt.find_clique_meets(set);
    }

    //! Returns an edge whose separator is a subset of the supplied
    //! set of variables.  \see junction_tree::find_separator_cover
    edge find_separator_cover(const domain_type& set) const {
      return jt.find_separator_cover(set);
    }

    //! Outputs the vertices whose cliques overlap the supplied set of nodes.
    template <typename OutIt>
    OutIt find_intersecting_cliques(const domain_type& set, OutIt out) const {
      return jt.find_intersecting_cliques(set, out);
    }

    //! Returns true if variables x are independent from variables y
    //! given variables z in this model.
    bool d_separated(const domain_type& x, const domain_type& y,
                     const domain_type& z = domain_type::empty_set) const {
      return jt.d_separated(x, y, z);
    }

    //! Returns the markov graph for the model
    sill::markov_graph<variable_type*> markov_graph() const {
      return jt.markov_graph();
    }

    //! Checks whether the cliques of this decomposable model are triangulated.
    void check_running_intersection() {
      jt.check_validity();
    }

    /**
     * Checks that this decomposable model is valid.  This method will
     * generate an assertion violation if the decomposable model is
     * not valid.  The junction tree must be valid, and the clique and
     * separator marginals must have the correct argument sets.
     *
     * \todo This method should also check for edge consistency.
     */
    void check_validity() const {
      // Check that the junction tree is valid.
      jt.check_validity();
      // Check that the arguments of clique & separator marginals match JT.
      foreach(vertex v, vertices())
        if (clique(v) != marginal(v).arguments()) {
          std::cerr << "check_validity() failed: "
                    << "clique(v) != marginal(v).arguments():"
                    << std::endl;
          std::cerr << "  clique(v) = " << clique(v)
                    << ", arguments() = " << marginal(v).arguments()
                    << std::endl;
          std::cerr << (*this);
          assert(false);
        }
      foreach(edge e, edges())
        if (separator(e) != marginal(e).arguments()) {
          std::cerr << "check_validity() failed: "
                    << "separator(e) != marginal(e).arguments():"
                    << std::endl;
          std::cerr << "  separator(e) = " << separator(e)
                    << ", arguments() = " << marginal(e).arguments()
                    << std::endl;
          std::cerr << (*this);
          assert(false);
        }
      // TODO: check that marginalizing the clique marginals to the
      // separator yields the separator marginal.
    }

    /**
     * Prints a human-readable representation of the decomposable
     * model to the supplied output stream.
     *
     * \todo Standardize with factors.
     */
    template <typename OutputStream>
    void print(OutputStream& out) const {
      jt.print(out);
    }

    operator std::string() const {
      std::ostringstream out;
      out << *this;
      return out.str();
    }

    // Probabilistic model queries
    //==========================================================================

    //! Number of arguments in this model.
    size_t num_arguments() const {
      return args.size();
    }

    //! returns domain rather than const domain& to implement factorized_model
    domain_type arguments() const {
      return args;
    }

    //! \todo this function has not been tested yet
    forward_range<const F&> factors() const {
      return std::make_pair(potential_iterator(this), potential_iterator());
    }

    //! Returns the clique marginal associated with a vertex.
    const F& marginal(vertex v) const {
      return jt[v];
    }

    //! Returns the separator marginal associated with an edge.
    const F& marginal(edge e) const {
      return jt[e];
    }

    //! Returns the clique marginals of this decomposable model
    forward_range<const F&> clique_marginals() const {
      return make_transformed(vertices(), vertex_property_functor(*this));
    }

    //! Computes a marginal over an arbitrary subset of variables.
    F marginal(const domain_type& vars) const {
      #ifdef SILL_VERBOSE
        std::cerr << "Computing marginal for: " << vars << std::endl;
      #endif
      assert(includes(args, vars));
      if (vars.empty()) return F(1);

      // Look for a separator that covers the variables.
      edge e = find_separator_cover(vars);
      if (e != edge()) return jt[e].marginal(vars);

      // Look for a clique that covers the variables.
      vertex v = find_clique_cover(vars);
      if (v) return jt[v].marginal(vars);

      // Compute the smallest subtree of the junction tree that
      // covers the variables of the new clique.
      jt.mark_subtree_cover(vars, false);
      #ifdef SILL_VERBOSE
        std::cerr << "After marking subtree cover: " << std::endl
                  << *this << std::endl;
      #endif

      // Collect the clique marginals in the subtree.
      std::vector<F> marginals;
      foreach(vertex v, vertices())
        if (jt.marked(v)) marginals.push_back(jt[v]);

      // For each separator marginal in the subtree, divide it out of
      // a clique marginal that subsumes it.
      //! \todo It would be slightly more efficient to traverse the tree and
      //!       immediately divide the separator into one of the cliques.
      foreach(edge e, edges()) {
        if (jt.marked(e)) {
          #if 0 // DEBUG
            assert(jt.marked(e.source()) && jt.marked(e.target()));
          #endif
          marginals.push_back(invert(jt[e]));
        }
      }
      // Now we want to compute the product of the factors in the
      // vector and sum out all variables we don't need.  Use variable
      // elimination.
      return variable_elimination(marginals, vars, sum_product<F>(),
                                  min_degree_strategy());
    }

    //! Computes a marginal over an arbitrary subset of variables,
    //! storing the result in output.
    void marginal(const domain_type& vars, F& output) const {
      #ifdef SILL_VERBOSE
        std::cerr << "Computing marginal for: " << vars << std::endl;
      #endif
      assert(includes(args, vars));
      if (vars.empty()) {
        output = F(1);
        return;
      }

      // Look for a separator that covers the variables.
      edge e = find_separator_cover(vars);
      if (e != edge()) {
        jt[e].marginal(vars, output);
        return;
      }

      // Look for a clique that covers the variables.
      vertex v = find_clique_cover(vars);
      if (v) {
        jt[v].marginal(vars, output);
        return;
      }

      // Compute the smallest subtree of the junction tree that
      // covers the variables of the new clique.
      jt.mark_subtree_cover(vars, false);
      #ifdef SILL_VERBOSE
        std::cerr << "After marking subtree cover: " << std::endl
                  << *this << std::endl;
      #endif

      // Collect the clique marginals in the subtree.
      std::vector<F> marginals;
      foreach(vertex v, vertices())
        if (jt.marked(v)) marginals.push_back(jt[v]);

      // For each separator marginal in the subtree, divide it out of
      // a clique marginal that subsumes it.
      //! \todo It would be slightly more efficient to traverse the tree and
      //!       immediately divide the separator into one of the cliques.
      foreach(edge e, edges()) {
        if (jt.marked(e)) {
          #if 0 // DEBUG
            assert(jt.marked(e.source()) && jt.marked(e.target()));
          #endif
          marginals.push_back(invert(jt[e]));
        }
      }
      // Now we want to compute the product of the factors in the
      // vector and sum out all variables we don't need.  Use variable
      // elimination.
      output = variable_elimination(marginals, vars, sum_product<F>(),
                                    min_degree_strategy());
    } // marginal()

    //! Computes a decomposable model that represents the marginal
    //! distribution over one ore more variables
    //! Note: This can potentially create enormous cliques via marginalization.
    void marginal(const domain_type& vars, decomposable& output) const {
      assert(includes(args, vars));
      output.jt.clear();
      output.args.clear();
      if (vars.empty()) return;

      // Create a model over a subtree covering 'vars'
      std::vector<F> orig_marginals;
      jt.mark_subtree_cover(vars, true);
      foreach(vertex v, vertices()) {
        if (jt.marked(v))
          orig_marginals.push_back(jt[v]);
      }
      foreach(edge e, edges()) {
        if (jt.marked(e))
          orig_marginals.push_back(invert(jt[e]));
      }
      // Sum out variables in the subtree which are not in 'vars'
      std::list<F> new_marginals;
      variable_elimination(orig_marginals, vars, sum_product<F>(),
                           min_degree_strategy(), new_marginals);
      output *= new_marginals;
    }

    /**
     * Computes the entropy of the distribution.
     *
     * @param base   Base of logarithm in entropy.
     * @return double representing the entropy of the distribution.
     */
    double entropy(double base) const {
      double result = 0;
      foreach(vertex v, vertices())
        result += marginal(v).entropy(base);
      foreach(edge e, edges())
        result -= marginal(e).entropy(base);
      return result;
    }

    /**
     * Computes the entropy of the distribution (using log base e).
     *
     * @return double representing the entropy of the distribution.
     */
    double entropy() const {
      return entropy(std::exp(1.));
    }

    /**
     * Computes the entropy H(X),
     * where X must be a subset of the arguments of this model.
     *
     * @param base   Base of logarithm in entropy.
     * @return double representing the entropy
     */
    double entropy(const domain_type& X, double base = std::exp(1.)) const {
      decomposable Xmodel;
      marginal(X, Xmodel);
      return Xmodel.entropy();
    }

    /**
     * Computes the conditional entropy H(Y | X),
     * where Y,X must be subsets of the arguments of this model.
     *
     * @param base   Base of logarithm in entropy.
     * @return double representing the conditional entropy
     *
     * @todo This could probably be done more efficiently.
     */
    double
    conditional_entropy(const domain_type& Y, const domain_type& X,
                        double base = std::exp(1.)) const {
      decomposable YXmodel;
      marginal(set_union(Y,X), YXmodel);
      double result(YXmodel.entropy());
      decomposable Xmodel;
      YXmodel.marginal(X, Xmodel);
      result -= Xmodel.entropy();
      return result;
    }

    /**
     * Approximates the conditional entropy H(Y | X) by sampling x ~ X
     * and computing H(Y | X=x).
     * This is useful if the marginalization required to compute H(Y|X) exactly
     * creates giant cliques, as long as the marginalization required to
     * compute H(Y | X=x) does not create giant cliques.
     * This decides it has converged when
     *  (standard error of estimate) / estimate < mult_std_error.
     *
     * @param base             Base of logarithm in entropy.
     * @param mult_std_error   Convergence parameter (> 0).
     * @return <approx conditional entropy, std error of estimate>
     */
    template <typename RandomNumberGenerator>
    std::pair<double, double>
    approx_conditional_entropy(const domain_type& Y, const domain_type& X,
                               double mult_std_error,
                               RandomNumberGenerator& rng,
                               double base = std::exp(1.)) const {
      assert(mult_std_error > 0);
      double val(0.); // accumulate sum
      double val2(0.); // accumulate sum of squares
      size_t APPROX_CHECK_PERIOD = 50;
      for (size_t i(0); i < std::numeric_limits<size_t>::max() - 1; ++i) {
        assignment_type a(sample(rng));
        decomposable Yx_model(*this);
        assignment_type ax;
        foreach(variable_type* v, X)
          ax[v] = a[v];
        Yx_model.condition(ax);
        decomposable Y_given_x_model;
        Yx_model.marginal(Y, Y_given_x_model);
        double result(Y_given_x_model.entropy());
        val += result;
        val2 += result * result;
        if ((i+1) % APPROX_CHECK_PERIOD == 0) {
          double est(val / (i+1.));
          double stderr_(sqrt((val2 / (i+1.)) - est * est)/sqrt(i+1.));
          if (stderr_ / est < mult_std_error)
            return std::make_pair(est, stderr_);
        }
      }
      throw std::runtime_error
        ("decomposable::approx_conditional_entropy() hit sample limit without getting a suitable approximation.");
    }

    /**
     * Computes the mutual information I(A; B),
     * where A,B must be subsets of the arguments of this model.
     * This is computed using I(A; B) = H(A) - H(A | B).
     *
     * @param base   Base of logarithm.
     * @return double representing the mutual information.
     */
    double
    mutual_information(const domain_type& A, const domain_type& B,
                       double base = std::exp(1.)) const {
      return entropy(A,base) - conditional_entropy(A,B,base);
    }

    /**
     * Computes the mutual information I(A; B),
     * where A,B must be subsets of the arguments of this model.
     * This is computed using I(A; B) = H(A) - H(A | B).
     * This uses approx_conditional_entropy() to compute H(A | B).
     *
     * @param mult_std_error   Convergence parameter (> 0).
     *                         (See approx_conditional_entropy().)
     * @param base   Base of logarithm.
     */
    template <typename RandomNumberGenerator>
    double
    approx_mutual_information(const domain_type& A, const domain_type& B,
                              double mult_std_error, RandomNumberGenerator& rng,
                              double base = std::exp(1.)) const {
      return entropy(A,base)
        - approx_conditional_entropy(A,B,mult_std_error,rng,base).first;
    }

    /**
     * Computes the conditional mutual information I(A; B | C),
     * where A,B,C must be subsets of the arguments of this model.
     * This is computed using I(A; B | C) = H(A | C) - H(A | B,C).
     *
     * @param base   Base of logarithm.
     * @return double representing the conditional mutual information.
     */
    double
    conditional_mutual_information(const domain_type& A, const domain_type& B,
                                   const domain_type& C,
                                   double base = std::exp(1.)) const {
      return conditional_entropy(A,C,base)
        - conditional_entropy(A,set_union(B,C),base);
    }

    /**
     * Computes the conditional mutual information I(A; B | C),
     * where A,B,C must be subsets of the arguments of this model.
     * This is computed using I(A; B | C) = H(A | C) - H(A | B,C).
     * This uses approx_conditional_entropy().
     *
     * @param mult_std_error   Convergence parameter (> 0).
     *                         (See approx_conditional_entropy().)
     * @param base   Base of logarithm.
     */
    template <typename RandomNumberGenerator>
    double
    approx_conditional_mutual_information(const domain_type& A,
                                          const domain_type& B,
                                          const domain_type& C,
                                          double mult_std_error,
                                          RandomNumberGenerator& rng,
                                          double base = std::exp(1.)) const {
      return approx_conditional_entropy(A,C,mult_std_error,rng,base).first
        - approx_conditional_entropy(A,set_union(B,C),mult_std_error,rng,base).first;
    }

    /**
     * Compute the max probability assignment.
     *
     * Note: This is virtual to support approx_decomposable (experimental).
     */
    virtual assignment_type max_prob_assignment() const {
      if (num_vertices() == 1) {
        // Handle special case with 1 vertex/factor.
        return arg_max(jt[*(vertices().first)]);
      }
      std::map<vertex, F> v2f;
      foreach(vertex v, vertices())
        v2f[v] = jt[v];
      flow_functor_mpa1 ffm1(v2f);
      assignment_type mpa;
      flow_functor_mpa2 ffm2(v2f, mpa);
      mpp_traversal(*this, ffm1, ffm2);
      return mpa;
    }

    //! Returns a sample from this model.
    template <typename RandomNumberGenerator>
    assignment_type sample(RandomNumberGenerator& rng) const {
      assignment_type a;
      flow_functor_sampling<RandomNumberGenerator> ffs(a, rng);
      ffs.sample_vertex(jt.root(), *this);
      pre_order_traversal(*this, jt.root(), ffs);
      return a;
    }

    // Losses
    //==========================================================================

    /**
     * Computes the log likelihood of the given assignment according to this
     * distribution.
     *
     * @param a    Assignment to a subset of the variables of the distribution.
     *             Note that if this assigns values to variables not in this
     *             model, then those variables are ignored.
     * @param base base of the log, default e
     * @return log likelihood of the assignment
     */
    double log_likelihood(const assignment_type& a, double base) const {
      using std::log;
      bool a_includes_args = true;
      foreach(variable_type* v, args) {
        if (a.find(v) == a.end()) {
          a_includes_args = false;
          break;
        }
      }
      if (a_includes_args) {
        double loglike = 0;
        foreach(vertex v, vertices())
          loglike += marginal(v).logv(a);
        foreach(edge e, edges())
          loglike -= marginal(e).logv(a);
        return (loglike / log(base));
      } else {
        decomposable marginal_model(*this);
        marginal_model.marginalize_out(set_difference(args, keys(a)));
        return marginal_model.log_likelihood(a, base);
      }
    }

    /**
     * Computes the log likelihood of the given assignment according to this
     * distribution.
     *
     * @param r    Record with values for a subset of the variables of the
     *             distribution.
     *             Note that if this assigns values to variables not in this
     *             model, then those variables are ignored.
     * @param base base of the log, default e
     * @return log likelihood of the assignment
     */
    double log_likelihood(const record_type& r, double base) const {
      using std::log;
      bool r_includes_args(true);
      foreach(variable_type* v, args) {
        if (!r.has_variable(v)) {
          r_includes_args = false;
          break;
        }
      }
      if (r_includes_args) {
        double loglike(0.);
        foreach(vertex v, vertices())
          loglike += marginal(v).logv(r);
        foreach(edge e, edges())
          loglike -= marginal(e).logv(r);
        return (loglike / log(base));
      } else {
        decomposable marginal_model(*this);
        marginal_model.marginalize_out(set_difference(args, r.variables()));
        return marginal_model.log_likelihood(r, base);
      }
    }

    double log_likelihood(const assignment_type& a) const {
      return log_likelihood(a, exp(double(1)));
    }

    double log_likelihood(const record_type& r) const {
      return log_likelihood(r, exp(double(1)));
    }

    //! Probability of the assignment P(a)
    logarithmic<double> operator()(const assignment_type& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    //! Probability of the assignment P(r)
    logarithmic<double> operator()(const record_type& r) const {
      return logarithmic<double>(log_likelihood(r), log_tag());
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected log likelihood E[log P(X)].
     */
    model_log_likelihood_functor<decomposable>
    log_likelihood(double base = exp(1.)) const {
      return model_log_likelihood_functor<decomposable>(*this, base);
    }

    /**
     * Computes the conditional log likelihood: log P(y|x),
     * where this distribution represents P(Y,X).
     *
     * @param X    Variables (which MUST be a subset of this model's arguments)
     *             to condition on.
     * @param base base of the log (default = e)
     *
     * @todo Add support for when this model represents P(Y,X,Z).
     */
    double conditional_log_likelihood(const record_type& r,
                                      const domain_type& X,
                                      double base = exp(1.)) const {
      decomposable tmp_model(*this);
      tmp_model.condition(r.assignment(X));
      return tmp_model.log_likelihood(r, base);
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected conditional log likelihood E[log P(Y|X)].
     */
    model_conditional_log_likelihood_functor<decomposable>
    conditional_log_likelihood(const domain_type& X,
                               double base = exp(1.)) const {
      return
        model_conditional_log_likelihood_functor<decomposable>(*this, X, base);
    }

    /**
     * Computes the per-label accuracy (average over X variables).
     * @param a    an assignment to this model's arguments
     */
    double per_label_accuracy(const assignment_type& a) const {
      double acc = 0;
      assignment_type pred(max_prob_assignment());
      foreach(variable_type* v, args) {
        if (equal(pred[v], safe_get(a, v)))
          ++acc;
      }
      return (acc / args.size());
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected per-label accuracy.
     */
    model_per_label_accuracy_functor<decomposable>
    per_label_accuracy() const {
      return model_per_label_accuracy_functor<decomposable>(*this);
    }

    /**
     * Computes the per-label accuracy of predicting Y given X,
     * where this distribution represents P(Y,X).
     * @param a    an assignment to this model's arguments
     * @param X    Variables (which MUST be a subset of this model's arguments)
     *             to condition on.
     */
    double
    per_label_accuracy(const assignment_type& a, const domain_type& X) const {
      assignment_type tmpa;
      foreach(variable_type* v, X) {
        tmpa[v] = safe_get(a, v);
      }
      decomposable tmp_model(*this);
      tmp_model.condition(tmpa);
      return tmp_model.per_label_accuracy(a);
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected per-label accuracy of predicting Y given X,
     * where this distribution represents P(Y,X).
     * @param X    Variables (which MUST be a subset of this model's arguments)
     *             to condition on.
     */
    model_per_label_accuracy_functor<decomposable>
    per_label_accuracy(const domain_type& X) const {
      return model_per_label_accuracy_functor<decomposable>(*this, X);
    }

    /**
     * Returns 1 if this predicts all variable values correctly and 0 otherwise.
     */
    size_t accuracy(const assignment_type& a) const {
      assignment_type pred(max_prob_assignment());
      foreach(variable_type* v, args) {
        if (!equal(pred[v],safe_get(a, v))) {
          return 0;
        }
      }
      return 1;
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected accuracy.
     */
    model_accuracy_functor<decomposable>
    accuracy() const {
      return model_accuracy_functor<decomposable>(*this);
    }

    /**
     * Computes the mean squared error (mean over variables).
     * Note: This is equivalent to per_label_accuracy for finite variables.
     *
     * @param a    an assignment to X
     */
    double mean_squared_error(const assignment_type& a) const {
      return (error_measures::squared_error<assignment_type>
              (a, max_prob_assignment(), args) / args.size());
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected mean squared error.
     */
    model_mean_squared_error_functor<decomposable>
    mean_squared_error() const {
      return model_mean_squared_error_functor<decomposable>(*this);
    }

    /**
     * Computes the mean squared error of predicting Y given X,
     * where this model is of P(Y,X).
     */
    double
    mean_squared_error(const assignment_type& a, const domain_type& X) const {
      assignment_type tmpa;
      foreach(variable_type* v, X) {
        tmpa[v] = safe_get(a, v);
      }
      decomposable tmp_model(*this);
      tmp_model.condition(tmpa);
      return (error_measures::squared_error<assignment_type>
              (a, tmp_model.max_prob_assignment(), args) / args.size());
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected mean squared error of predicting Y given X.
     */
    model_mean_squared_error_functor<decomposable>
    mean_squared_error(const domain_type& X) const {
      return model_mean_squared_error_functor<decomposable>(*this, X);
    }

    // Restructuring operations
    //==========================================================================

    /**
     * Restructures this decomposable model so that it includes the
     * supplied cliques.  These cliques can include new variables
     * (which are not current arguments); in this case, the cliques
     * and separators containing them will have marginals over only
     * the current variables.
     *
     * Note: this operation invalidates all iterators and descriptors.
     *
     * @param cliques An input range over the cliques that should be
     *        added to the model.
     *
     * \todo test this code
     */
    template <typename CliqueRange>
    void add_cliques(const CliqueRange& cliques) {
      concept_assert((InputRangeConvertible<CliqueRange, domain_type>));
      /* TODO: right now we retriangulate the entire model, but only
         the subtree containing the parameter vars must be
         retriangulated */

      // Compute the new set of arguments.
      domain_type new_args = args;

      foreach(const domain_type& clique, cliques) {
//        new_args = set_union(new_args, clique);
        new_args.insert(clique.begin(), clique.end());
      }

      // Create a graph with the cliques of this decomposable model.
      sill::markov_graph<variable_type*> mg(new_args);
      foreach(vertex v, vertices())
        mg.add_clique(clique(v));
      foreach(const domain_type& clique, cliques)
        mg.add_clique(clique);

      // Now create a new junction tree for the Markov graph and
      // initialize the clique/separator marginals
      jt_type new_jt(mg, min_degree_strategy());
      foreach(vertex v, new_jt.vertices()) {
        const domain_type& clique = new_jt.clique(v);
        new_jt[v] = marginal(set_intersect(clique, arguments()));
      }

      foreach(edge e, new_jt.edges()) {
        const domain_type& separator = new_jt.separator(e);
        new_jt[e] = marginal(set_intersect(separator,arguments()));
      }

      // Install the new tree and update the arguments
      jt.swap(new_jt);
      args.swap(new_args);
    }

    /**
     * Restructures this decomposable model so that it has a clique
     * that covers the supplied variables, and returns the vertex
     * associated with this clique.  The caller should first check
     * that no such cover exists by calling #find_clique_cover on the
     * junction tree.
     *
     * @param vars
     *        The set of variables for which a cover should be
     *        created.  This set of variables should be a subset of
     *        the current arguments of this decomposable model.
     */
    vertex make_cover(const domain_type& vars) {
      add_cliques(boost::make_iterator_range(&vars, &vars+1));
      // Now find a cover in the new junction tree.
      return find_clique_cover(vars);
    }

    /**
     * Merges two vertices in the junction tree.  This operation
     * swings all edges from the source of the supplied edge to the
     * target.  The source is removed from the graph.
     */
    void merge(edge e) {
      vertex u = e.source();
      vertex v = e.target();

      // Get the cliques and marginals incident to the edge.  Store
      // them by value, because altering the junction tree below will
      // remove their storage.
      domain_type clique_u = clique(u);
      domain_type clique_v = clique(v);
      F marginal_u = marginal(u);
      F marginal_s = marginal(e);

      // Update the junction tree. This operation removes u and retains v
      jt.merge(e);
      if (clique(v) == clique_u)
        jt[v] = marginal_u;
      else if (clique(v) == clique_v)
        ;// The marginal at v was retained during merge
      else {
        jt[v] /= marginal_s;
        jt[v] *= marginal_u;
      }
    }

    /**
     * Removes a vertex from the junction tree if it is nonmaximal.
     * @return true if a subsuming neighbor was found.
     */
    bool remove_if_nonmaximal(vertex u) {
      // Look for a neighbor of this vertex whose clique subsumes this
      // vertex's clique.
      foreach(edge e, out_edges(u)) {
        vertex v = e.target();
        if (includes(clique(v), clique(u))) {
          merge(e);
          return true;
        }
      }
      return false;
    }

    //! Clears all factors and variables from this model.
    void clear() {
      args.clear();
      jt.clear();
    }

    // Distribution updates
    //==========================================================================

    /**
     * Multiplies the supplied collection of factors into this
     * decomposable model and renormalizes it.
     *
     * @param factors A readable forward range of objects of type F
     */
    template <typename Range>
    decomposable& operator*=(const Range& factors)
    {
      concept_assert((ReadableForwardRangeConvertible<Range, F>));

      // Retriangulate the model so that it contains a clique for each factor.
      add_cliques(make_transformed(factors, arguments_functor<F>()));

      // For each factor, multiply it into a clique that subsumes it.
      // We do not use F for iteration since Range may be over a different
      // factor type that is merely convertible to F.
      foreach(const typename Range::value_type& factor, factors) {
        if (!factor.arguments().empty()) {
          vertex v = find_clique_cover(factor.arguments());
          assert(v != vertex());
          jt[v] *= factor;
          /*
           // It is sometimes OK for gaussians to be unnormalizable
           // before calibration.
          if (!jt[v].is_normalizable()) {
            std::cerr << "Cannot normalize clique potential after absorbing "
                      << "factor.  Original potential:" << std::endl
                      // << orig_potential << std::endl
                      << "New factor: " << std::endl
                      << factor << std::endl
                      << "Unnormalizable result: " << std::endl
                      << jt[v] << std::endl;
            throw normalization_error("decomposable::operator*= ran into factor which could not be normalized.");
          }
          */
        }
      }

      // Depending on the factor type, pre-normalize clique marginals
      // to avoid numerical issues.
      if (impl::decomposable_extra_normalization<F>::value) {
        foreach(const vertex& v, vertices()) {
          if (jt[v].is_normalizable())
            jt[v].normalize();
        }
      }

      // Recalibrate and renormalize the model.
      calibrate();

      normalize();
      return *this;
    }

    /**
     * Multiplies the supplied factor into this decomposable model and
     * renormalizes the model.
     */
    decomposable& operator*=(const F& factor) {
      return (*this) *= boost::make_iterator_range(&factor, &factor + 1);
    }

    /**
     * This replaces all current factors with the given factors
     * while keeping the current structure of this model.  Each new factor
     * must be covered by a clique in this model.
     *
     * If you are calling this multiple times,
     * it is significantly faster to use the other replace_factors() method.
     *
     * @param factors A readable forward range of objects of type F
     */
    template <typename Range>
    void replace_factors(const Range& factors) {
      std::vector<vertex> factor_vertex_map;
      foreach(const typename Range::value_type& factor, factors) {
        vertex v(find_clique_cover(factor.arguments()));
        assert(v != vertex());
        factor_vertex_map.push_back(v);
      }
      replace_factors(factors, factor_vertex_map);
    }

    /**
     * This replaces all current factors with the given factors
     * while keeping the current structure of this model.  Each new factor
     * must be covered by a clique in this model.
     *
     * @param factors            A readable forward range of objects of type F
     * @param factor_vertex_map  A vector corresponding to the given factor
     *                            range, with each entry being a vertex in this
     *                            model whose clique covers the corresponding
     *                            factor's arguments.
     */
    template <typename Range>
    void replace_factors(const Range& factors,
                         const std::vector<vertex>& factor_vertex_map) {
      concept_assert((ReadableForwardRangeConvertible<Range, F>));

      foreach(const edge& e, edges())
        jt[e] = F(1);
      foreach(const vertex& v, vertices())
        jt[v] = F(1);

      // We do not use F for iteration since Range may be over a different
      // factor type that is merely convertible to F.
      size_t j(0);
      foreach(const typename Range::value_type& factor, factors) {
        if (!factor.arguments().empty()) {
          assert(j < factor_vertex_map.size());
          const vertex& v = factor_vertex_map[j];
          jt[v] *= factor;
          /*
           // It is sometimes OK for gaussians to be unnormalizable
           // before calibration.
          if (!jt[v].is_normalizable()) {
            std::cerr << "Cannot normalize this factor:\n"
                      << factor << std::endl;
            throw normalization_error("decomposable::replace_factors ran into factor which could not be normalized.");
          }
          */
        }
        ++j;
      }

      // Depending on the factor type, pre-normalize clique marginals
      // to avoid numerical issues.
      if (impl::decomposable_extra_normalization<F>::value) {
        foreach(const vertex& v, vertices()) {
          if (jt[v].is_normalizable())
            jt[v].normalize();
        }
      }

      // Recalibrate and renormalize the model.
      calibrate();
      normalize();
    }

    /**
     * Conditions this decomposable model on an assignment to one or
     * more of its variables. This is a mutable operation.
     *
     * @param a
     *        An assignment to some variables.  This assignment is
     *        instantiated in each clique and separator factor, and
     *        the distribution is subsequently normalized to yield
     *        the conditional distribution given the assignment as
     *        evidence.
     */
    decomposable& condition(const assignment_type& a) {
      // Compute the variables that are conditioned on.
      domain_type restricted_vars(set_intersect(keys(a), arguments()));
      if (restricted_vars.empty())
        return *this;

      // Find all cliques that contain an old variable.
      typename std::vector<vertex> vertices;
      find_intersecting_cliques(restricted_vars, std::back_inserter(vertices));

      // Update each affected clique and separator
      foreach(vertex v, vertices) {
        jt[v] = jt[v].restrict(a);
        if (jt[v].is_normalizable()) {
          jt[v].normalize();
        } else {
          std::cerr << "Cannot normalize this factor:\n" << jt[v] << std::endl;
            throw normalization_error("decomposable::condition ran into factor which could not be normalized.");
        }
        foreach(edge e, out_edges(v)) {
          if (set_intersect(jt[e].arguments(), restricted_vars).size() > 0) {
            jt[e] = jt[e].restrict(a);
            if (jt[e].is_normalizable()) {
              jt[e].normalize();
            } else {
              std::cerr << "Cannot normalize this factor:\n"
                        << jt[e] << std::endl;
              throw normalization_error("decomposable::condition ran into factor which could not be normalized.");
            }
          }
        }
        jt.set_clique(v, set_difference(clique(v), restricted_vars));
      }

      // Update the arguments & recalibrate.
      args = set_difference(args, restricted_vars);
      calibrate();
      normalize();
      return *this;
    }

    F restrict_flatten(const assignment_type& a) const {
      F result(1.0);
      foreach(vertex v, jt.vertices()) {
        result *= jt[v].restrict(a);
      }
      foreach(edge e, jt.edges()) {
        result /= jt[e].restrict(a);
      }
      return result;
    }

    /**
     * Marginalizes a set of variables out of this decomposable model.
     *
     * For each variable, we find a subtree that includes the variable
     * and contracts all the edges in that subtree.
     *
     * @param  vars
     *         the variables to be marginalized out
     *
     * \todo do we want to allow vars to have variables other than args
     *       of this factor?
     * \todo The implementation somewhat inefficient at the moment.
     */
    void marginalize_out(const domain_type& vars) {
      #ifdef SILL_VERBOSE
        std::cerr << "Marginalizing out: " << vars << std::endl;
      #endif
      // Marginalize each variable out independently.
      foreach(variable_type* var, vars) {
        while (true) {
          // Find a cover for this variable.
          vertex v = find_clique_cover(make_domain(var));
          assert(v != vertex());
          // Look for a neighbor that also has the variable.
          bool done = true;
          foreach(edge e, out_edges(v)) {
            vertex w = e.target();
            if (clique(w).count(var)) {
              // The cliques at v and w both contain the variable.
              // Merge them and then restart this process.
              done = false;
              merge(e);
              break;
            }
          }
          if (done) {
            // None of the cliques neighboring v contain the variable.
            // By the running intersection property, then, no other
            // cliques contain the variable.  So we can safely
            // marginalize the variable out of this clique, and then
            // we are done.
            jt.set_clique(v, set_difference(clique(v), make_domain(var)));
            jt[v] = jt[v].marginal(clique(v));
            remove_if_nonmaximal(v);
            // Move on to the next variable to marginalize out.
            break;
          }
        }
      }

      // Update the arguments.
      args = set_difference(args, vars);
      #ifdef SILL_VERBOSE
        std::cerr << "Result: " << *this << std::endl;
      #endif
    }

    /**
     * Marginalizes a set of variables out of this decomposable model
     * using an approximation.  The variables are marginalized out of
     * each clique independently; since clique merging is avoided,
     * this operation is not exact.
     *
     * @param  vars
     *         the variables to be marginalized out
     *
     * \todo At the moment, the implementation is somewhat inefficient.
     */
    void marginalize_out_approx(const domain_type& vars) {
      #ifdef SILL_VERBOSE
        std::cerr << "Marginalizing out (approx): " << vars << std::endl;
      #endif
      std::vector<vertex> overlapping;
      while (true) {
        // Find a vertex whose clique overlaps vars.
        overlapping.clear();
        find_intersecting_cliques(vars, std::back_inserter(overlapping));
        if (overlapping.empty()) break;
        vertex v = overlapping.front();
        // Remove vars from this clique.
        jt.set_clique(v, set_difference(clique(v), vars));
        // at this point, the variables have been removed from the separators
        jt[v] =  jt[v].marginal(clique(v));

        // Marginalize out the variables from all incident separators.
        foreach(edge e, out_edges(v)) {
          //! \todo In a way, this check should not be necessary
          //!       These optimizations should be performed inside factors
          domain_type sep_args = jt[e].arguments();
          if (set_intersect(sep_args, vars).size() > 0)
            jt[e] = sum(jt[e], vars);
        }
        remove_if_nonmaximal(v);
      }
      // Update the arguments.
      args.erase(vars.begin(), vars.end());
      #ifdef SILL_VERBOSE
        std::cerr << "Result: " << *this << std::endl;
      #endif
    }


    /**
     * Renames (a subset of) the arguments of this factor.
     *
     * @param map
     *        an object such that map[v] maps the variable handle v
     *        a type compatible variable handle; this mapping must be 1:1.
     */
    void subst_args(const std::map<variable_type*, variable_type*>& var_map) {
      // Compute the variables to be replaced.
      domain_type old_vars = keys(var_map);

      // Find all cliques that contain an old variable.
      std::vector<vertex> vertices;
      find_intersecting_cliques(old_vars, std::back_inserter(vertices));

      // Update affected cliques and incident separator marginals
      foreach(vertex v, vertices) {
        jt.set_clique(v, subst_vars(clique(v), var_map));
        jt[v].subst_args(var_map);
        foreach(edge e, out_edges(v))
          jt[e].subst_args(var_map);
      }

      // Update the arguments.
      args = subst_vars(args, var_map);
    }

    /**
     * Applies the supplied functor to all clique marginals of this
     * decomposable model, and then recalibrates the model.
     *
     * \todo should also apply to separator marginals???
     * \todo do we ever need this???
     * \todo Should this also normalize the model?
     */
    template <typename Functor>
    void apply(Functor f) {
      //! \todo define a concept for "Updater"
      foreach(vertex v, vertices()) jt[v].apply(f);
      calibrate();
    }

    // Protected member functions
    //==========================================================================
  protected:

    //! Returns a reference to the clique potential associated with a vertex.
    factor_type& potential(vertex v) {
      return jt[v];
    }

    //! Returns a reference to the separator potential associated with an edge
    factor_type& potential(edge e) {
      return jt[e];
    }

    /**
     * Passes flows outwards from the supplied vertex.
     *
     * The current implementation uses a breadth-first search,
     * passing a flow along each edge as it is added to the search tree.
     *
     * Note: This is virtual to support approx_decomposable (experimental).
     */
    virtual void distribute_evidence(vertex v) {
      flow_functor flow_func;
      flow_func.require_normalizable = true;
      pre_order_traversal(*this, v, flow_func);
    }

    /**
     * Recalibrates the model by passing flows using the message
     * passing protocol.
     *
     * Note: This is virtual to support approx_decomposable (experimental).
     */
    virtual void calibrate() {
      flow_functor flow_func_post;
      flow_func_post.require_normalizable =
        impl::decomposable_extra_normalization<F>::value;
      flow_functor flow_func_pre;
      flow_func_pre.require_normalizable = true;
      mpp_traversal(*this, flow_func_post, flow_func_pre);
    }

    /**
     * Normalizes this decomposable model; all clique and separator
     * marginals are normalized.
     *
     * Note: This is virtual to support approx_decomposable (experimental).
     */
    virtual void normalize() {
      foreach(vertex v, vertices())
        jt[v].normalize();
      foreach(edge e, edges())
        jt[e].normalize();
    }

    /**
     * Reorder factor variables to support more efficient calibration.
     *  - Start with the root node; keep its variables in the current order.
     *  - Do a search from the root node; at each edge u --> v,
     *     - Order the separator vars in the same order as in u.
     *     - Order v's vars with the separator vars first (as the least
     *       significant vars) and the new vars second.
     *     - Note: For models with cliques of size > 2, the separator's vars
     *       may not be contiguous in u's var order;
     *       i.e., u may have order (A,B,C) while the separator has order (A,C).
     * This ordering means that, during calibration,
     *  - During the post-order traversal, we go from v to u along edge e:
     *     - Divide [u] /= [e] (and multiply analogously)
     *        - RIGHT HERE NOW
     *     - Get marginal over [e] of [v].
     *  - During the pre-order traversal,
     */

    // Protected member class definitions
    //==========================================================================
  protected:

    /**
     * A functor which passes flows through the junction tree.
     * Given an edge e = (u, v), this function passes flow from u to v.
     * This version requires node and edge factors to be normalizable after
     * the flow has passed.
     */
    struct flow_functor {

      //! If true, require that node and edge factors be normalizable after
      //! the flow has passed.
      //!  (default = true)
      bool require_normalizable;

      flow_functor()
        : require_normalizable(true) { }

      void operator()(edge e, decomposable& dm) {
        // Get the source and target vertices.
        vertex u = e.source();
        vertex v = e.target();

        // Compute the new separator potential using u and
        // update v's potential with ratio of the new and the old separator
        dm.jt[v] /= dm.jt[e];
        dm.jt[u].marginal_unnormalized(dm.separator(e), dm.jt[e]);
        dm.jt[v] *= dm.jt[e];

        if (require_normalizable) {
          if (!dm.jt[e].is_normalizable() ||
              !dm.jt[v].is_normalizable()) {
            std::cerr << "Cannot normalize after flow.  "
                      << "Source potential:\n" << dm.jt[u] << "\n"
                      << "separator potential:\n" << dm.jt[e] << "\n"
                      << "target potential:\n" << dm.jt[v] << std::endl;
            throw normalization_error
              (std::string("decomposable::flow_functor::operator()") +
               " ran into factor which could not be normalized.");
          }
        }
      }
    }; // struct flow_functor

    /**
     * A functor which passes flows through the junction tree for computing
     * max probability assignments.
     * Given an edge e = (u, v), this function passes flow from u to v
     * during the post-order traversal.
     * \todo Make sure this is correct!
     */
    struct flow_functor_mpa1 {
      std::map<vertex, F>& v2f;

      //! v2f must be initialized with the clique and edge potentials
      flow_functor_mpa1(std::map<vertex, F>& v2f)
        : v2f(v2f) {
      }

      void operator()(edge e, const decomposable& dm) {
        using std::endl;
        // Get the source and target vertices.
        vertex u = e.source();
        vertex v = e.target();

        // General case: compute the new separator mpa assignment using u and
        // update v's mpa assignment with ratio of the new and the old values.
        // i.e., f_v <-- max_{clique(u) \ separator(e)} f_u f_v / f_e
        v2f[v] *= v2f[u].maximum(dm.separator(e));
        v2f[v] /= dm.marginal(e);
        v2f[v].normalize(); // Helps with numerical issues.

        // DEBUG
        // Note: this would fail for Gaussian factors which may not be
        // initially normalizable; add a flag
        if (!v2f[v].is_normalizable()) {
          std::cerr << "Cannot normalize after flow.  Source potential:"
                    << endl << v2f[u] << endl
                    << "separator potential: " << endl
                    << dm.marginal(e) << endl
                    << "target potential: " << endl
                    << v2f[v] << endl;
          throw normalization_error
            (std::string("decomposable::flow_functor_mpa::operator()") +
             " ran into factor which could not be normalized.");
        }
      }
    };  // struct flow_functor_mpa1

    /**
     * A functor which passes flows through the junction tree for computing
     * max probability assignments.
     * Given an edge e = (u, v), this function passes flow from u to v
     * during the pre-order traversal.
     * \todo Make sure this is correct!
     */
    struct flow_functor_mpa2 {
      std::map<vertex, F>& v2f;
      assignment_type& mpa;

      /**
       * @param v2f   must be same map which was given to flow_functor_mpa1
       * @param mpa   This must be empty, and it will hold the MPA at the end.
       */
      flow_functor_mpa2(std::map<vertex, F>& v2f, assignment_type& mpa)
        : v2f(v2f), mpa(mpa) {
      }

      void operator()(edge e, const decomposable& dm) {
        // For root: mpa will be empty, so compute MPA of clique(root).
        if (mpa.empty()) {
          mpa = arg_max(v2f[e.source()]);
        }

        // Set values in v which are not already set.
        assignment_type mpa_v(arg_max(v2f[e.target()].restrict(mpa)));
        mpa.insert(mpa_v.begin(), mpa_v.end());
      }
    };  // struct flow_functor_mpa2

    /**
     * An iterator over the factors of this decomposable model.
     * For clique marginals, returns a reference to the factor.
     * For separator marginals, returns a reference to a temporary
     * that holds the inverted marginals.
     *
     * Implementation note: the fields are required to hold
     * valid content only if stage != END.
     */
    class potential_iterator
    : public std::iterator<std::forward_iterator_tag, const F> {

      //! The current iteration stage (vertices, edges, end)
      enum stage_type { VERTICES, EDGES, END } stage;

      //! The current vertex iterator and the end of vertex range
      vertex_iterator vit, vend;

      //! The current edge iterator and the end of edge range
      edge_iterator eit, eend;

      //! (A pointer to) the decomposable model being iterated over
      const decomposable* dm;

      //! The temporary used for inverted separator
      F sep_potential;

    public:
      //! Constructs an iterator over all vertices and edges of dm
      explicit potential_iterator(const decomposable* dm) : dm(dm) {
        if (dm->empty()) {
          stage = END;
        } else {
          boost::tie(vit, vend) = dm->vertices();
          boost::tie(eit, eend) = dm->edges();
          stage = VERTICES;
          assert(vit != vend);
        }
      }

      //! Constructs an iterator that represents the end of iteration range
      potential_iterator() {
        stage = END;
      }

      //! Prefix increment
      potential_iterator& operator++() {
        switch(stage) {
          case VERTICES:
            ++vit;
            if (vit == vend) stage = (eit == eend) ? END : EDGES;
            break;
          case EDGES:
            ++eit;
            if (eit == eend) stage = END;
            break;
          case END:
            assert(false);
            // once the iterator reaches the end, it cannot be incremented
        }
        if (stage == EDGES) // update the edge potential
          sep_potential = F(1) / dm->marginal(*eit);
        return *this;
      }

      //! Postfix increment (creates a temporary)
      potential_iterator operator++(int) {
        potential_iterator temp(*this);
        operator++();
        return temp;
      }

      //! Returns a reference to the current potential
      const F& operator*() const {
        switch (stage) {
          case VERTICES: return dm->marginal(*vit);
          case EDGES: return sep_potential;
          default: assert(false); /* at the end */ return *(F*)NULL;
        }
      }

      //! Returns true if two iterators are equal
      //! This function is guaranteed to work only for iterators that were
      //! invoked on the same decomposable model.
      bool operator==(const potential_iterator& other) const {
        if (stage == END || other.stage == END)
          return stage == other.stage;
        else
          return stage == other.stage && vit == other.vit && eit == other.eit;
      }

      //! Returns true if the two iteratore are not equal
      //! This function is guaranteed to work only for iterators that were
      //! invoked on the same decomposable model.
      bool operator!=(const potential_iterator& other) const {
        return !(*this == other);
      }
    }; // class potential_iterator

    /**
     * Functor for sampling from a decomposable model.
     */
    template <typename RandomNumberGenerator>
    struct flow_functor_sampling {

      //! Current assignment to variables.
      assignment_type& a;

      //! Random number generator
      RandomNumberGenerator& rng;

      flow_functor_sampling(assignment_type& a, RandomNumberGenerator& rng)
        : a(a), rng(rng) { }

      //! This should be called on the start vertex of the post order
      //! traversal after the traversal.
      void sample_vertex(const vertex& v, const decomposable& dm) {
        F f(dm[v].restrict(a));
        if (f.size() == 0)
          return;
        if (f.is_normalizable()) {
          f.normalize();
        } else {
          std::cerr << "unnormalizable factor: " << f << std::endl;
          throw normalization_error("decomposable::flow_functor_sampling::sample_vertex() ran into factor which could not be normalized.");
        }
        assignment_type tmpa(f.sample(rng));
        for (typename assignment_type::const_iterator tmpa_it(tmpa.begin());
             tmpa_it != tmpa.end(); ++tmpa_it)
          a[tmpa_it->first] = tmpa_it->second;
      }

      //! When applied to an edge, this samples the unsampled variables in
      //! the source vertex.
      void operator()(edge e, const decomposable& dm) {
        sample_vertex(e.target(), dm);
      }

    }; // struct flow_functor_sampling

  }; // class decomposable

  /**
   * Prints a human-readable representation of the decomposable model
   * to the supplied output stream.
   */
  template <typename F>
  std::ostream& operator<<(std::ostream& out, const decomposable<F>& model) {
    model.print(out);
    return out;
  }

} // namespace sill

namespace boost {

  //! A traits class that lets decomposable_model work in BGL algorithms
  template <typename F>
  struct graph_traits< sill::decomposable<F> >
    : public graph_traits<sill::junction_tree<typename F::variable_type*,F,F> >
  { };

} // namespace boost

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DECOMPOSABLE_HPP

