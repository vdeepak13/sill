#ifndef SILL_LEARNING_PWL_CRF_LEARNER_HPP
#define SILL_LEARNING_PWL_CRF_LEARNER_HPP

#include <set>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
//#include <sill/factor/concepts.hpp>
#include <sill/iterator/subset_iterator.hpp>
#include <sill/learning/crf/pwl_crf_weights.hpp>
#include <sill/learning/dataset_old/dataset_view.hpp>
#include <sill/model/crf_model.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for pwl_crf_learner.
   *
   * @tparam FactorType  factor type which fits the LearnableCRFfactor concept
   */
  template <typename FactorType>
  struct pwl_crf_learner_parameters
    : public pwl_crf_weights_parameters<FactorType> {

    //    concept_assert((sill::LearnableCRFfactor<FactorType>));

    typedef pwl_crf_weights_parameters<FactorType> base;

    //! If true, the learned model will be a tree or forest.
    //! It will be learned immediately, without the need to call add_edge().
    //!  (default = true)
    bool learn_tree;

    //! Regularization: penalty >= 0 for adding an edge.
    //!  (default = 0)
    //! @todo IMPLEMENT THIS.  Be careful about how we do the MST.
    double edge_reg;

    //! If true, retain the full edge_part_map instead of deleting it after
    //! the model is built.
    //!  (default = false)
    bool retain_edge_part_map;

    /**
     * Debugging modes:
     *  - 0: no debugging (default)
     *  - 1: print progress through functions
     *  - 2: print factor scores as they are computed in init()
     *  - above: same as highest debugging mode
     */
    size_t DEBUG;

    pwl_crf_learner_parameters()
      : base(), learn_tree(true), edge_reg(0), retain_edge_part_map(false),
        DEBUG(0) { }

    bool valid() const {
      if (!base::valid())
        return false;
      if (edge_reg < 0)
        return false;
      if (edge_reg != 0)
        return false; // B/C NOT IMPLEMENTED YET
      return true;
    }

  }; // class pwl_crf_learner_parameters

  /**
   * Class for learning the structure of a CRF P(Y|X) using piecewise
   * likelihood.
   *
   * About piecewise likelihood:
   *  - It is based on a bound for the partition function, and
   *    it allows you to get a decomposable score which permits you to
   *    learn a tree CRF via a simple max spanning tree algorithm.
   *  - See "Piecewise training of undirected models"
   *    by C Sutton, A McCallum (2005).
   *
   * @tparam FactorType  factor type which fits the LearnableCRFfactor concept
   *
   * \author Joseph Bradley
   * \ingroup learning_structure
   * @todo This currently only learns pairwise models, but we could support
   *       higher-order factors.
   */
  template <typename FactorType>
  class pwl_crf_learner {

    //    concept_assert((sill::LearnableCRFfactor<FactorType>));

    // Public classes
    //==========================================================================
  public:

    //! CRF factor type
    typedef FactorType crf_factor;

    //! Type of CRF model being learned
    typedef crf_model<crf_factor> crf_model_type;

    //! Type of output variable Y.
    typedef typename crf_model_type::output_variable_type output_variable_type;

    //! Type of input variable X.
    typedef typename crf_model_type::input_variable_type input_variable_type;

    //! Type of variable for both Y,X.
    typedef typename crf_model_type::variable_type variable_type;

    //! Type of domain for variables in Y.
    typedef typename crf_model_type::output_domain_type output_domain_type;

    //! Type of domain for variables in X.
    typedef typename crf_model_type::input_domain_type input_domain_type;

    //! Type of domain for variables in both Y,X.
    typedef typename crf_model_type::domain_type domain_type;

    //! Type of assignment for variables in Y.
    typedef typename crf_model_type::output_assignment_type
      output_assignment_type;

    //! Type of assignment for variables in X.
    typedef typename crf_model_type::input_assignment_type
      input_assignment_type;

    //! Type of assignment for variables in both Y,X.
    typedef typename crf_model_type::assignment_type assignment_type;

    /**
     * The type which this factor f(Y,X) outputs to represent f(Y, X=x).
     * For finite Y, this will probably be table_factor;
     * for vector Y, this will probably be gaussian_factor.
     */
    typedef typename crf_model_type::output_factor_type output_factor_type;

    //! CRF factor regularization type.
    typedef typename crf_factor::regularization_type crf_factor_reg_type;

    //! Parameters type.
    typedef pwl_crf_learner_parameters<crf_factor> parameters;

    // Protected classes
    //==========================================================================
  protected:

    friend class mst_weight_functor;
    friend class mst_edge_inserter;

    //! Type of graph used for MST algorithm.
    //! Vertices correspond to Y variables; each edge stores the edge score.
    typedef undirected_graph<output_variable_type*, void_, double> ig_type;

    //! Functor used by the MST algorithm.
    //! This returns the negation of the edge scores.
    class mst_weight_functor {
    public:
      typedef typename ig_type::edge argument_type;
      typedef double result_type;
    private:
      const ig_type* g;
    public:
      mst_weight_functor(const ig_type& g) : g(&g) { }
      double operator()(typename ig_type::edge e) const {
        return g->operator[](e);
      }
    };

    //! Functor used by the MST algorithm.
    //! This is used to construct the returned tree.
    class mst_edge_inserter
      : public std::iterator<std::output_iterator_tag, void, void, void, void> {

    public:
      typedef typename ig_type::edge argument_type;

      //! Total weight of edges in the CRF.
      double total_score;

    private:
      //! The CRF model
      crf_model<crf_factor>* g;

      //! The graph used for the MST algorithm
      const ig_type* ig_graph;

      //! The pwl_crf_learner
      const pwl_crf_learner* crf_learner_ptr;

    public:
      mst_edge_inserter()
        : total_score(0), g(NULL), ig_graph(NULL), crf_learner_ptr(NULL) { }

      mst_edge_inserter(crf_model<crf_factor>* g, const ig_type& ig_graph,
                        const pwl_crf_learner& crf_learner)
        : total_score(0), g(g), ig_graph(&ig_graph),
          crf_learner_ptr(&crf_learner) { }

      //! Assignment
      mst_edge_inserter& operator=(const typename ig_type::edge& e) {
        assert(g);
        assert(ig_graph);
        // Insert edge e (from the ig_type graph) into the CRF graph.
        output_domain_type Yvars;
        Yvars.insert(e.source());
        Yvars.insert(e.target());
        std::pair<output_variable_type*,output_variable_type*>
          Ypair(e.source() < e.target() ?
                std::make_pair(e.source(), e.target()) :
                std::make_pair(e.target(), e.source()));
        g->add_factor
          (safe_get(crf_learner_ptr->weight_functor_.edge_part_map(),Ypair));
        total_score -= ig_graph->operator[](e);
        return *this;
      }

      //! Does nothing
      mst_edge_inserter& operator*() {
        return *this;
      }

      //! Does nothing.
      mst_edge_inserter& operator++() {
        return *this;
      }

      //! Does nothing
      mst_edge_inserter operator++(int) {
        return *this;
      }
    }; // class mst_edge_inserter

    // Protected data members
    //==========================================================================

    parameters params;

    //! Functor for computing the edge weights.
    pwl_crf_weights<crf_factor> weight_functor_;

    //! Current CRF factor graph.
    crf_model<crf_factor> model_;

    //! (For learning non-tree models)
    //! Queue of factors to add: <factor arguments in Y, score>
    mutable_queue<output_domain_type, double> factor_queue;

    //! Sum of scores from all factors added to the model (not counting
    //! single-variable factors).
    double total_score_;

    // Protected methods
    //==========================================================================

    /**
     * Initialize stuff.
     * If learning a tree, go ahead and learn it.
     * Otherwise, fill queue with potential edges and their scores.
     */
    void init(const dataset<>& ds,
              const output_domain_type& Yvars) {
      assert(params.valid());
      boost::mt11213b rng(params.random_seed);

      if (params.DEBUG > 0) {
        std::cerr << "pwl_crf_learner::init(): computing all edge scores:"
                  << std::endl;
      }
      if (params.learn_tree) {
        double total_time(0);
        boost::timer timer;
        size_t nedges_done(0);
        // Construct a graph for the MST algorithm, with Y variables for
        // vertices and edges marked with their scores.
        ig_type mst_graph;
        foreach(output_variable_type* v, Yvars)
          mst_graph.add_vertex(v);
        subset_iterator<output_domain_type> Yvars_end;
        for (subset_iterator<output_domain_type> Yvars_it(Yvars, 2);
             Yvars_it != Yvars_end; ++Yvars_it) {
          const output_domain_type& twoYvars = *Yvars_it;
          typename output_domain_type::const_iterator
            twoYvars_it(twoYvars.begin());
          assert(twoYvars_it != twoYvars.end());
          output_variable_type* Yvar1 = *twoYvars_it;
          ++twoYvars_it;
          assert(twoYvars_it != twoYvars.end());
          output_variable_type* Yvar2 = *twoYvars_it;
          timer.restart();
          double score_(weight_functor_(Yvar1, Yvar2).first);
          total_time += timer.elapsed();
          mst_graph.add_edge(Yvar1, Yvar2, - score_);
          if (params.DEBUG > 1) {
            ++nedges_done;
            std::cerr << "  avg time: " << (total_time/nedges_done) << "; "
                      << twoYvars << ": " << score_ << std::endl;
          }
        }
        mst_edge_inserter mst_e_i(&model_, mst_graph, *this);
        kruskal_minimum_spanning_tree(mst_graph, mst_e_i,
                                      mst_weight_functor(mst_graph));
        total_score_ = mst_e_i.total_score;
        if (!params.retain_edge_part_map)
          weight_functor_.clear_edge_part_map();
      } else {
        // Add a single-variable factor for each output variable.
        foreach(output_variable_type* fv, Yvars) {
          output_domain_type tmpdom(make_domain<output_variable_type>(fv));
          crf_factor f;
          if (params.crf_factor_cv) {
            f =
              learn_crf_factor<crf_factor>::train_cv
              (params.cv_params,
               ds, tmpdom, weight_functor_.X_mapping()[tmpdom],
               *(params.crf_factor_params_ptr),
               boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
          } else {
            f =
              learn_crf_factor<crf_factor>::train
              (ds, tmpdom, weight_functor_.X_mapping()[tmpdom],
               *(params.crf_factor_params_ptr),
               boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
          }
          model_.add_factor(f);
        }
        // Compute edge scores and factors.
        double total_time(0);
        boost::timer timer;
        size_t nedges_done(0);
        subset_iterator<output_domain_type> Yvars_end;
        for (subset_iterator<output_domain_type> Yvars_it(Yvars, 2);
             Yvars_it != Yvars_end; ++Yvars_it) {
          const output_domain_type& twoYvars = *Yvars_it;
          typename output_domain_type::const_iterator
            twoYvars_it(twoYvars.begin());
          assert(twoYvars_it != twoYvars.end());
          output_variable_type* Yvar1 = *twoYvars_it;
          ++twoYvars_it;
          assert(twoYvars_it != twoYvars.end());
          output_variable_type* Yvar2 = *twoYvars_it;
          timer.restart();
          double score_(weight_functor_(Yvar1, Yvar2).first);
          total_time += timer.elapsed();
          factor_queue.push(twoYvars, score_);
          if (params.DEBUG > 1) {
            ++nedges_done;
            std::cerr << "  avg time: " << (total_time/nedges_done) << "; "
                      << twoYvars << ": " << score_ << std::endl;
          }
        }
      }
    } // init()

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Constructor for a learner for a model for P(Y | X).
     * This initially holds an empty graph.
     *
     * @param ds            Training dataset.
     * @param Yvars         Output variables.
     * @param X_mapping     Map specifying which X vars to use for each factor.
     *                      Note this type supports automatic conversions from
     *                      various types of input mappings.
     * @param parameters    algorithm parameters
     */
    pwl_crf_learner
    (const dataset<>& ds, const output_domain_type& Yvars,
     const crf_X_mapping<crf_factor>& X_mapping,
     parameters params = parameters())
      : params(params), weight_functor_(ds, Yvars, X_mapping, 0, params),
        total_score_(0) {
      init(ds, Yvars);
    }

    // Getters and helper methods
    //==========================================================================

    //! Returns the current model structure.
    const typename crf_model<crf_factor>::crf_graph_type&
    current_graph() const {
      return model_;
    }

    //! Returns the current model, parametrized using piecewise likelihood.
    const crf_model<crf_factor>& current_model() const {
      return model_;
    }

    //! Returns the sum of scores from all factors added to the model
    //! (not counting single-variable factors).
    double total_score() const {
      return total_score_;
    }

    // Learning and mutating methods
    //==========================================================================

    /**
     * Adds one more factors to the model, or none if the objective or
     * parameters prevent it.
     * @return  The domain in Y of the factor which was added, or an empty
     *          domain if no factor was added.
     */
    output_domain_type add_factor() {
      if (params.learn_tree || factor_queue.empty())
        return output_domain_type();
      std::pair<output_domain_type, double> top_factor(factor_queue.pop());
      model_.add_factor(safe_get(weight_functor_.edge_part_map(),
                                 top_factor.first));
      total_score_ += top_factor.second;
      model_.simplify_unary(top_factor.first);
      return top_factor.first;
    }

    /**
     * Choose regularization parameters for structure learning.
     * This is a heuristic which seems reasonable and is much faster than
     * doing actual cross-validation.  This does the following:
     *  - Given a dataset and the CRF factor type (as a template parameter),
     *  - For n_folds times,
     *     - Choose a random pair of variables in Y.
     *     - Train a CRF factor to compute P(Y | X_Y) for each possible lambda.
     *  - See which lambda did best according to log likelihood.
     * Note: For more info on this method's parameters, see the constructors for
     *       this class.
     *
     * @param means      (Return value.) Means of scores for the given lambdas.
     * @param stderrs    (Return value.) Std errors of scores for the lambdas.
     * @param reg_params Regularization parameters to try.
     * @param n_folds    Number of cross-validation folds (in (0, dataset size])
     * @param params     Parameters for the CRF factor type.
     * @param random_seed  This uses this random seed, not the one in the
     *                     algorithm parameters.
     * @return  Chosen regularization parameters.
     */
    static crf_factor_reg_type
    choose_lambda
    (vec& means, vec& stderrs,
     const std::vector<crf_factor_reg_type>& reg_params,
     size_t n_folds, const dataset<>& ds, const output_domain_type& Yvars,
     const crf_X_mapping<crf_factor>& X_mapping,
     const typename crf_factor::parameters& params,
     unsigned random_seed = time(NULL)) {

      assert((n_folds > 0) && (n_folds <= ds.size()));
      assert(reg_params.size() > 0);
      means.zeros(reg_params.size());
      stderrs.zeros(reg_params.size());
      boost::mt11213b rng(random_seed);
      std::vector<output_variable_type*> Yvector(Yvars.begin(), Yvars.end());
      dataset_view<> permuted_view(ds);
      permuted_view.set_record_indices(randperm(ds.size(), rng));
      typename crf_factor::parameters tmp_params(params);
      dataset_view<> fold_train_view(permuted_view);
      dataset_view<> fold_test_view(permuted_view);
      fold_train_view.save_record_view();
      fold_test_view.save_record_view();
      // For each fold
      for (size_t fold(0); fold < n_folds; ++fold) {
        // Prepare the fold dataset views
        if (fold != 0) {
          fold_train_view.restore_record_view();
          fold_test_view.restore_record_view();
        }
        fold_train_view.set_cross_validation_fold(fold, n_folds, false);
        fold_test_view.set_cross_validation_fold(fold, n_folds, true);
        // Pick a random pair of Y variables.
        output_domain_type tmpYset;
        std::vector<size_t> tmpYset_indices(randperm(Yvars.size(), rng, 2));
        foreach(size_t i, tmpYset_indices)
          tmpYset.insert(Yvector[i]);
        for (size_t k(0); k < reg_params.size(); ++k) {
          tmp_params.reg = reg_params[k];
          crf_factor tmpf =
            learn_crf_factor<crf_factor>::train
            (fold_train_view, tmpYset, X_mapping[tmpYset], tmp_params,
             boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
          double tmpval(tmpf.log_expected_value(fold_test_view));
          means[k] += tmpval;
          stderrs[k] += tmpval * tmpval;
        }
      }
      foreach(double& val, means)
        val /= n_folds;
      for (size_t k(0); k < means.size(); ++k)
        stderrs[k] = std::sqrt((stderrs[k] / n_folds) - (means[k] * means[k]));
      return reg_params[max_index(means, rng)];

    } // choose_lambda()

    /**
     * Choose regularization parameters for structure learning.
     * This is a fancy version of the above method which tries out more values
     * for lambda.
     * Specifically, it does the following:
     *  - It tries a grid of values and picks the best one.
     *  - It then tries a grid of new values around the best lambda found so far
     *    (in the range denoted by the lambdas bracketing the best lambda).
     *
     * @param reg_params (Return value.) Parameters which were tried.
     * @param means      (Return value.) Means of scores for the given lambdas.
     * @param stderrs    (Return value.) Std errors of scores for the lambdas.
     * @param cv_params  Parameters specifying how to do cross validation.
     * @param params     Parameters for the CRF factor type.
     * @param random_seed  This uses this random seed, not the one in the
     *                     algorithm parameters.
     * @return  Chosen regularization parameters.
     */
    static crf_factor_reg_type
    choose_lambda_fancy
    (std::vector<crf_factor_reg_type>& reg_params, vec& means, vec& stderrs,
     const crossval_parameters& cv_params,
     const dataset<>& ds, const output_domain_type& Yvars,
     const crf_X_mapping<crf_factor>& X_mapping,
     const typename crf_factor::parameters& params,
     unsigned random_seed = time(NULL)) {

      assert(cv_params.valid());
      boost::mt11213b rng(random_seed);
      boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
      std::vector<crf_factor_reg_type> reg_params1;
      std::vector<vec>
        reg_lambdas(create_parameter_grid(cv_params.minvals, cv_params.maxvals,
                                          cv_params.nvals,cv_params.log_scale));
      foreach(const vec& v, reg_lambdas) {
        crf_factor_reg_type tmprp;
        tmprp.lambdas = v;
        reg_params1.push_back(tmprp);
      }
      vec means1, stderrs1;
      crf_factor_reg_type best_reg_param1 =
        choose_lambda(means1, stderrs1, reg_params1, cv_params.nfolds,
                      ds, Yvars, X_mapping, params, unif_int(rng));
      if (cv_params.zoom > 1)
        assert(false); // TO DO
      if (cv_params.zoom == 0) {
        reg_params = reg_params1;
        means = means1;
        stderrs = stderrs1;
        return best_reg_param1;
      }
      reg_lambdas = zoom_parameter_grid(reg_lambdas, best_reg_param1.lambdas,
                                        cv_params.nvals, cv_params.log_scale);
      std::vector<crf_factor_reg_type> reg_params2;
      foreach(const vec& v, reg_lambdas) {
        crf_factor_reg_type tmprp;
        tmprp.lambdas = v;
        reg_params2.push_back(tmprp);
      }
      vec means2, stderrs2;
      crf_factor_reg_type best_reg_param2 =
        choose_lambda(means2, stderrs2, reg_params2, cv_params.nfolds,
                      ds, Yvars, X_mapping, params, unif_int(rng));
      size_t best_i1(0);
      for (size_t i(0); i < reg_params1.size(); ++i) {
        if ((reg_params1[i].regularization == best_reg_param1.regularization)
            && equal(reg_params1[i].lambdas, best_reg_param1.lambdas)) {
          best_i1 = i;
          break;
        }
      }
      size_t best_i2(0);
      for (size_t i(0); i < reg_params2.size(); ++i) {
        if ((reg_params2[i].regularization == best_reg_param2.regularization)
            && equal(reg_params2[i].lambdas, best_reg_param2.lambdas)) {
          best_i2 = i;
          break;
        }
      }
      reg_params = concat(reg_params1, reg_params2);
      means = concat(means1, means2);
      stderrs = concat(stderrs1, stderrs2);
      if (means1[best_i1] >= means2[best_i2])
        return best_reg_param1;
      else
        return best_reg_param2;

    } // choose_lambda_fancy()

    //! Returns a const reference to the weight functor.
    //! This is mainly useful for retreiving stored info maps.
    const pwl_crf_weights<crf_factor>& weight_functor() const {
      return weight_functor_;
    }

  }; // class pwl_crf_learner

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_PWL_CRF_LEARNER_HPP
