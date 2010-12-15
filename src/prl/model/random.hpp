#ifndef PRL_RANDOM_MODEL
#define PRL_RANDOM_MODEL

#include <prl/global.hpp>

#include <prl/range/concepts.hpp>
#include <prl/factor/gaussian_crf_factor.hpp>
#include <prl/factor/random.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/factor/table_crf_factor.hpp>
#include <prl/model/bayesian_network.hpp>
#include <prl/model/crf_model.hpp>
#include <prl/model/decomposable.hpp>
#include <prl/model/markov_network.hpp>
#include <prl/base/stl_util.hpp>
#include <prl/base/universe.hpp>

#include <prl/macros_def.hpp>

/**
 * \file random.hpp  Construct synthetic examples of common graphical models.
 *
 * File contents:
 *  - Methods for generative models
 *     - randomize_factors
 *     - random_ising_model
 *     - random_HMM
 *  - Methods for conditional models over discrete variables
 *     - create_fancy_random_crf
 *     - create_random_crf
 *     - create_random_chain_crf
 *  - Methods for conditional models over vector variables
 *     - create_random_gaussian_crf
 *     - create_chain_gaussian_crf
 *  - Auxiliary methods for working with generated models
 *     - create_analogous_generative_model
 */

namespace prl {

  //! \addtogroup model
  //! @{

  // Methods for generative models
  //============================================================================

  /*
  template <typename F, typename Range>
  pairwise_markov_network<F>
  make_grid_mrf(size_t m, size_t n, const Range& variables) {
    concept_assert((InputRandomAccessRangeC<Range, variable_h>));
    concept_assert((Factor<F>));
    assert(variables.size() == m*n);
    return pairwise_markov_network<F>(make_grid_graph(m, n), variables);
  }
  */

  template <typename F, typename Engine>
  void randomize_factors(pairwise_markov_network<F>& mn, Engine& engine) {
    foreach(F& factor, mn.factors())
      factor = random_discrete_factor<F>(factor.arguments(), engine);
  }

  template <typename F, typename Engine>
  void random_ising_model(pairwise_markov_network<F>& mn, Engine& engine) {
    typedef typename F::variable_type variable_type;

    foreach(variable_type* v, mn.vertices()) {
      finite_domain f; f.insert(v);
      mn[v] = random_discrete_factor<F>(f, engine);
    }

    foreach(undirected_edge<variable_type*> e, mn.edges()) {
      mn[e] = random_ising_factor<F>(e.source(), e.target(), engine);
    }
  }

  template <typename Engine>
  void random_ising_model(double alpha_max, double w_max,
                          pairwise_markov_network<table_factor>& mn, Engine& engine) {
    boost::uniform_real<> unif_alpha(0, alpha_max);
    boost::uniform_real<> unif_w(0, w_max);
    foreach(finite_variable* v, mn.vertices()) {
      mn[v] = make_ising_factor<table_factor>(v, unif_alpha(engine));
    }
    foreach(undirected_edge<finite_variable*> e, mn.edges()) {
      mn[e] = make_ising_factor<table_factor>(e.source(), e.target(), unif_w(engine));
    }
  }

  /**
   * Generates a random HMM as follows:
   *  - creates an HMM with n emitting states + 1 initial state which
   *    emits nothing
   *  - chooses transition probabilities from a Dirichlet(alpha_tr)
   *  - chooses emission probabilities from a Dirichlet(alpha_em)
   * This clears the given Bayes net and creates new variables.
   * 
   * @param n        number of time states, excluding initial state
   * @param n_states number of states (for each hidden node)
   * @param n_emissions  number of emissions (for each observed node)
   * @param alpha_tr Dirichlet parameter for choosing transition probabilities
   *                 (and initial state probabilities)
   * @param alpha_em Dirichlet parameter for choosing emission probabilities
   * @return  <hidden variables, emission variables>, in order;
   *          Note there is 1 more hidden variable than there are emission.
   */
  template <typename F, typename Engine>
  std::pair<std::vector<typename F::variable_type*>,
            std::vector<typename F::variable_type*> >
  random_HMM(bayesian_network<F>& bn, Engine& engine, universe& u,
             size_t n, size_t n_states, size_t n_emissions,
             double alpha_tr = 1, double alpha_em = 1) {
    bn.clear();
    typedef typename F::domain_type domain_type;
    typedef typename F::variable_type variable_type;
    std::vector<variable_type*> hidden_vars;
    std::vector<variable_type*> emission_vars;
    variable_type* prev_v = u.new_finite_variable(n_states);
    hidden_vars.push_back(prev_v);
    F init_fctr(random_discrete_conditional_factor<table_factor>
                (make_domain(prev_v), domain_type(), alpha_tr, engine));
    bn.add_factor(prev_v, init_fctr);
    for (size_t i = 0; i < n; ++i) {
      variable_type* cur_v = u.new_finite_variable(n_states);
      F cur_fctr(random_discrete_conditional_factor<table_factor>
                 (make_domain(cur_v), make_domain(prev_v), alpha_tr, engine));
      bn.add_factor(cur_v, cur_fctr);
      variable_type* emit_v = u.new_finite_variable(n_emissions);
      F emit_fctr(random_discrete_conditional_factor<table_factor>
                  (make_domain(emit_v), make_domain(cur_v), alpha_em, engine));
      bn.add_factor(emit_v, emit_fctr);
      prev_v = cur_v;
      hidden_vars.push_back(prev_v);
      emission_vars.push_back(emit_v);
    }
    return std::make_pair(hidden_vars, emission_vars);
  }

  // Methods for conditional models over discrete variables
  //============================================================================

  /**
   * Creates models useful for doing tests with CRFs P(Y|X).
   * This generates a decomposable model for P(X) and a CRF for P(Y|X),
   * where each of P(X), P(Y|X) can be chains or trees.
   *
   * This may be used to create tractable or intractable models,
   * depending on how you use the resulting P(X), P(Y|X) and on the 'tractable'
   * parameter:
   *  - For tractable P(Y,X): Set the 'tractable' parameter to true, and
   *    use model_product() to create a model Q(Y,X) from P(X) and P(Y|X).
   *    Note that Q(Y,X) != P(X)P(Y|X) in general.  Sample from the tractable
   *    joint Q(Y,X).
   *  - For intractable P(Y,X): Set the 'tractable' parameter to true (in which
   *    case P(X), P(Y|X) are both chains/trees following the same structure)
   *    or to false (in which case P(X), P(Y|X) follow different structures).
   *    Sample x ~ P(X) and then sample y ~ P(Y|X=x).  Note the normalization
   *    constants make the joint model intractable in general.
   *
   * To explain how models are generated, first look at the tractable Q(Y,X)
   * case for chains:
   *  - Q(Y,X) is a chain which forms a ladder structure.
   *     - e.g.: Y1--Y2--Y3--...
   *             |   |   |
   *             X1--X2--X3--...
   *  - So Q(Y|X) is also a chain:
   *     - e.g.: Y1--Y2--Y3--...
   *             |   |   |
   *             X1  X2  X3  ...
   *  - Note that Q(X) is not a chain; it is a giant clique.
   * For trees, this generates models analogously, but the Y-Y factors form a
   * tree instead of a chain.
   *  - The tree structure is built incrementally by attaching new Y variables
   *    to Y variables already in the tree with equal probability.
   *    (This is called 'non-preferential random attachment.')
   * For the intractable case, P(X) is built in a way analogous to that
   * described above for chains/trees, and then P(Y|X) is too.  Each Y still
   * corresponds to a single X, but this correspondence does not match their
   * structures.  Instead, the correspondence is chosen via a random
   * permutation of X.
   * 
   * The potentials may be generated as follows (in log space):
   *  - "random": Choose each value according to Uniform[-s,s].
   *       (s >= 0)
   *  - "associative": [s,0;0,s] (or diagonal equivalent for higher arities)
   *       (s in Reals)
   *  - "random_assoc": [s',0;0,s'], where s' ~ base + Uniform[-s,s]
   *       (s >= 0, base in Reals)
   * This permits controlled variability in potentials via alternating
   * default potentials with special potentials.  Specifically, different
   * parameters are given for the default and special potentials, and
   * another parameter tells this method how often to generate a special
   * potential.
   *
   * This returns a mapping from Y variables to sets of X variables.
   *  - For tractable models, each Y maps to its single corresponding X.
   *  - For intractable models, each Y maps to its Markov blanket in X.
   *
   * @param YXmodel       (Return value) Decomposable model for P(Y,X)
   * @param YgivenXmodel  (Return value) CRF model for P(Y|X)
   * @param n             number of Y variables (and X variables)
   * @param arity         arity of the variables
   * @param model_choice  "chain" or "tree"
   * @param tractable     If true, P(Y,X) will be tractable.
   * @param factor_choice "random" / "associative" / "random_assoc"
   * @param YYstrengthD   Value 's' for Y-Y default potentials; see above.
   * @param YXstrengthD   Value 's' for Y-X default potentials; see above.
   * @param XXstrengthD   Value 's' for X-X default potentials; see above.
   * @param strength_baseD     Used for factor_choice = "random_assoc" (for
   *                           default potentials).
   * @param YYstrengthS   Value 's' for Y-Y special potentials; see above.
   * @param YXstrengthS   Value 's' for Y-X special potentials; see above.
   * @param XXstrengthS   Value 's' for X-X special potentials; see above.
   * @param strength_baseS     Used for factor_choice = "random_assoc" (for
   *                           special potentials).
   * @param factor_periods     Periods for inserting special factors for
   *                           [YY, YX, XX]. If this is k, then the k^th factor
   *                           will be special, etc. (with indexing from 0).
   *                           If this is 0, then all will be special.
   * @param add_cross_factors  If true, add factors (Y_i, X_{i+1}), etc.
   *
   * @return <Y variables in order, X variables in order,
   *          mapping from Y variables to their corresponding X variables>
   */
  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
  create_fancy_random_crf(decomposable<table_factor>& Xmodel,
                          crf_model<table_crf_factor>& YgivenXmodel,
                          size_t n, size_t arity, universe& u,
                          const std::string& model_choice, bool tractable,
                          const std::string& factor_choice,
                          double YYstrengthD, double YXstrengthD,
                          double XXstrengthD, double strength_baseD,
                          double YYstrengthS, double YXstrengthS,
                          double XXstrengthS, double strength_baseS,
                          ivec factor_periods, bool add_cross_factors,
                          unsigned random_seed);

  /**
   * Creates models useful for doing tests with CRFs P(Y|X).
   * This generates a decomposable model for P(X) and a CRF for P(Y|X).
   * NOTE: You may multiply the factors together to get a model Q(Y,X),
   *       but Q(Y,X) != P(X) P(Y|X) in general.
   * It can generate chains and trees.
   * See create_fancy_random_crf() for structure and parametrization details.
   *
   * @param Xmodel       empty decomposable model for P(X)
   * @param YgivenXmodel  empty CRF model for P(Y|X)
   * @param n             number of Y variables (and X variables)
   * @param arity         arity of the variables
   * @param model_choice  "chain" or "tree"
   * @param factor_choice "random" / "associative" / "random_assoc"
   * @param YYstrength    Value 's' for Y-Y potentials; see above.
   * @param YXstrength    Value 's' for Y-X potentials; see above.
   * @param XXstrength    Value 's' for X-X potentials; see above.
   * @param add_cross_factors   If true, add factors (Y_i, X_{i+1}), etc.
   * @param strength_base Used for factor_choice = "random_assoc"
   * @return <Y variables in order, X variables in order,
   *          mapping from Y variables to their Markov blankets in X>
   */
  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
  create_random_crf(decomposable<table_factor>& Xmodel,
                    crf_model<table_crf_factor>& YgivenXmodel,
                    size_t n, size_t arity, universe& u,
                    const std::string& model_choice,
                    const std::string& factor_choice,
                    double YYstrength, double YXstrength,
                    double XXstrength, bool add_cross_factors,
                    unsigned random_seed,
                    double strength_base = 0);

  /**
   * Easy-to-use version of create_random_crf() with fewer options,
   * useful for doing quick tests with CRFs P(Y|X).
   * This creates chain models with binary variables and random factors.
   * This generates a decomposable model for P(X) and a CRF for P(Y|X).
   * NOTE: You may multiply the factors together to get a model Q(Y,X),
   *       but Q(Y,X) != P(X) P(Y|X) in general.
   *
   * @param Xmodel        (return value) empty decomposable model for P(X)
   * @param YgivenXmodel  (return value) empty CRF model for P(Y|X)
   * @param n             number of Y variables (and X variables)
   * @return <Y variables in order, X variables in order,
   *          mapping from Y variables to their Markov blankets in X>
   */
  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
  create_random_chain_crf(decomposable<table_factor>& Xmodel,
                          crf_model<table_crf_factor>& YgivenXmodel,
                          size_t n, universe& u,
                          unsigned random_seed = time(NULL));

  // Methods for conditional models over vector variables
  //============================================================================

  /**
   * Creates CRFs P(Y|X) over vector variables (of 1 element each)
   * useful for doing tests.
   * This generates a decomposable model for P(X) and a CRF for P(Y|X).
   * It can generate chains and trees.
   *
   * For chains, it generates models as follows:
   *  - P(Y,X) is a chain which forms a ladder structure.
   *     - e.g.: Y1--Y2--Y3--...
   *             |   |   |
   *             X1--X2--X3--...
   *  - So P(Y|X) is also a chain:
   *     - e.g.: Y1--Y2--Y3--...
   *             |   |   |
   *             X1  X2  X3  ...
   *  - Note that P(X) is not a chain; it is a giant clique.
   *  - The potentials are generated using make_binary_marginal_gaussian().
   *
   * For trees, it generates models analogously, but the Y-Y factors form a tree
   * instead of a chain.
   *  - The tree structure is built incrementally by attaching new Y variables
   *    to Y variables already in the tree with equal probability.
   *
   * @param Xmodel        empty decomposable model for P(X)
   * @param YgivenXmodel  empty CRF model for P(Y|X)
   * @param n             number of Y variables (and X variables)
   * @param model_choice  "chain" or "tree"
   * @param b_max         Used to parametrize factors.
   * @param c_max         Used to parametrize factors.
   * @param variance      Used to parametrize factors.
   * @param YYcorrelation Used to parametrize factors.
   * @param YXcorrelation Used to parametrize factors.
   * @param XXcorrelation Used to parametrize factors.
   * @param add_cross_factors   If true, add factors (Y_i, X_{i+1}), etc.
   * @return <Y variables in order, X variables in order,
   *          mapping from Y variables to their Markov blankets in X>
   */
  boost::tuple<vector_var_vector, vector_var_vector,
               std::map<vector_variable*, copy_ptr<vector_domain> > >
  create_random_gaussian_crf(decomposable<canonical_gaussian>& Xmodel,
                             crf_model<gaussian_crf_factor>& YgivenXmodel,
                             size_t n, universe& u,
                             const std::string& model_choice,
                             double b_max, double c_max, double variance,
                             double YYcorrelation, double YXcorrelation,
                             double XXcorrelation,
                             bool add_cross_factors, unsigned random_seed);

  /**
   * Easy-to-use version of create_random_gaussian_crf() with fewer options,
   * useful for doing quick tests with CRFs P(Y|X).
   * This creates chain models with binary variables.
   * It generates a decomposable model for P(Y,X) and a CRF for P(Y|X).
   *
   * @param YXmodel       empty decomposable model for P(Y,X)
   * @param YgivenXmodel  empty CRF model for P(Y|X)
   * @param n             number of Y variables (and X variables)
   * @return <Y variables in order, X variables in order,
   *          mapping from Y variables to their Markov blankets in X>
   */
  boost::tuple<vector_var_vector, vector_var_vector,
               std::map<vector_variable*, copy_ptr<vector_domain> > >
  create_chain_gaussian_crf(decomposable<canonical_gaussian>& YXmodel,
                            crf_model<gaussian_crf_factor>& YgivenXmodel,
                            size_t n, universe& u, unsigned random_seed);

  // Auxiliary methods for working with generated models
  //============================================================================

  /**
   * Given a CRF structure and a mapping from input Y variables in the CRF to
   * output X variables, return a structure for a generative model which
   * is analogous to the CRF, i.e.:
   *  - For each clique among Y variables in the CRF, add the clique to the
   *    generative model.
   *  - For each y in Y in the CRF, add the clique (y, Y2X_map[y]) to the
   *    generative model.
   *
   * @param generative_structure   (Return value) Generative structure analogous
   *                               to the CRF structure.
   *                               Any previous contents are cleared.
   */
  template <typename GenerativeGraphType, typename CRFGraphType>
  void create_analogous_generative_structure
  (GenerativeGraphType& generative_structure,
   const CRFGraphType& crf_structure,
   const std::map<typename CRFGraphType::output_variable_type*, copy_ptr<typename CRFGraphType::input_domain_type> >& Y2X_map) {

    typedef typename CRFGraphType::output_variable_type output_variable_type;
    typedef typename CRFGraphType::input_variable_type input_variable_type;
    typedef typename CRFGraphType::variable_type variable_type;

    generative_structure.clear();
    undirected_graph<variable_type*> tmp_structure;
    foreach(const typename CRFGraphType::vertex& v,
            crf_structure.factor_vertices()) {
      tmp_structure.make_clique(crf_structure.output_arguments(v));
    }
    foreach(output_variable_type* var, crf_structure.output_arguments()) {
      tmp_structure.make_clique
        (set_union(make_domain(var), *(safe_get(Y2X_map, var))));
    }
    generative_structure.initialize(tmp_structure, min_degree_strategy());

  } // create_analogous_generative_model

  //! @}

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // PRL_RANDOM_MODEL
