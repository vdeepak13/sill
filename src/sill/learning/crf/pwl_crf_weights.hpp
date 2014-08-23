#ifndef SILL_PWL_CRF_WEIGHTS_HPP
#define SILL_PWL_CRF_WEIGHTS_HPP

#include <set>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
//#include <sill/factor/concepts.hpp>
#include <sill/iterator/subset_iterator.hpp>
#include <sill/learning/crf/crf_X_mapping.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/crf/learn_crf_factor.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for pwl_crf_weights.
   * This is designed so that classes which want to compute these weights
   * can have their parameter structs inherit from this parameter struct.
   *
   * @tparam FactorType  factor type which fits the LearnableCRFfactor concept
   */
  template <typename FactorType>
  struct pwl_crf_weights_parameters {

    //    concept_assert((sill::LearnableCRFfactor<FactorType>));

    //! CRF factor regularization type.
    typedef typename FactorType::regularization_type crf_factor_reg_type;

    /**
     * Score type:
     *  - 0: piecewise likelihood
     *       log( P(Y1,Y2 | X1,X2) )
     *  - 1: Discriminative Conditional Influence:
     *       log( P(Y1,Y2 | X1,X2) / [ P(Y1 | X1) P(Y2 | X2) ] )
     *       Note: This is best when using local evidence.
     *     (default)
     *  - 2: Conditional Mutual Information:
     *       log( P(Y1,Y2 | X1,X2) / [ P(Y1 | X1,X2) P(Y2 | X1,X2) ] )
     * Note: Conditional mutual information using global inputs, as well
     *       as mutual information ignoring the inputs, may be run using
     *       DCI/CMI by setting the crf_X_mapping appropriately.
     */
    size_t score_type;

    /**
     * CRF factor parameters specifying how the crf_factor type
     * calculates the conditional probabilities in the score.
     * Note that this contains any regularization parameters as well
     * but is overriden if crf_factor_choose_lambda is selected.
     *  (default = factor parameter defaults (if they exist!))
     */
    boost::shared_ptr<typename FactorType::parameters> crf_factor_params_ptr;

    //! If true, do cross-validation to choose the regularization parameters
    //! for pairwise factors, using the below parameters n_folds, minvals,
    //! maxvals, nlambdas, and zoom.
    //!  (default = false)
    bool crf_factor_cv;

    //! Parameters specifying how to do cross validation.
    crossval_parameters cv_params;

    //! If true, record score info for edges for which weights are computed.
    //!  (default = false)
    bool retain_edge_score_info;

    //! If true, record vertex_part_map (regardless of learning mode).
    //!  (default = false)
    bool retain_vertex_part_map;

    /**
     * Used to make the algorithm deterministic
     *    (default = time)
     */
    unsigned random_seed;

    pwl_crf_weights_parameters()
      : score_type(1), crf_factor_cv(false), retain_edge_score_info(false),
        retain_vertex_part_map(false),
        random_seed(time(NULL)) { }

    bool valid() const {
      if (score_type > 2)
        return false;
      if (!crf_factor_params_ptr)
        return false;
      if (!crf_factor_params_ptr->valid())
        return false;
      if (crf_factor_cv) {
        if (!cv_params.valid())
          return false;
      }
      return true;
    }

  }; // class pwl_crf_weights_parameters

  /**
   * Class for computing edge weights for CRF structure learning using
   * weights based on the piecewise likelihood:
   *  - piecewise likelihood (not a good option by itself)
   *  - Decomposable Conditional Influence (DCI)
   *  - Conditional Mutual Information (CMI) (with local inputs)
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
   * \ingroup learning_discriminative
   * @todo If the crf_X_mapping returns the same set X for all inputs, then
   *       use CMI instead of DCI since they should get the same results.
   * @todo This currently only supports edge weights, but we could support
   *       higher-order factors.
   * @todo The templated mode could be more efficient if we computed all
   *       edge scores at once, rather than doing N^2 lookups in the map
   *       of pre-computed regressors.
   */
  template <typename FactorType>
  class pwl_crf_weights {

    //    concept_assert((sill::LearnableCRFfactor<FactorType>));

    // Public classes
    //==========================================================================
  public:

    //! CRF factor type
    typedef FactorType crf_factor;

    //! Type of output variable Y.
    typedef typename crf_factor::output_variable_type output_variable_type;

    //! Type of input variable X.
    typedef typename crf_factor::input_variable_type input_variable_type;

    //! Type of variable for both Y,X.
    typedef typename crf_factor::variable_type variable_type;

    //! Type of domain for variables in Y.
    typedef typename crf_factor::output_domain_type output_domain_type;

    //! Type of domain for variables in X.
    typedef typename crf_factor::input_domain_type input_domain_type;

    //! Type of domain for variables in both Y,X.
    typedef typename crf_factor::domain_type domain_type;

    //! Type of assignment for variables in Y.
    typedef typename crf_factor::output_assignment_type output_assignment_type;

    //! Type of assignment for variables in X.
    typedef typename crf_factor::input_assignment_type input_assignment_type;

    //! Type of assignment for variables in both Y,X.
    typedef typename crf_factor::assignment_type assignment_type;

    //! Type of record for variables in Y.
    typedef typename crf_factor::output_record_type output_record_type;

    //! Type of record for variables in X.
    typedef typename crf_factor::input_record_type input_record_type;

    //! Type of record for variables in both Y,X.
    typedef typename crf_factor::record_type record_type;

    /**
     * The type which this factor f(Y,X) outputs to represent f(Y, X=x).
     * For finite Y, this will probably be table_factor;
     * for vector Y, this will probably be gaussian_factor.
     */
    typedef typename crf_factor::output_factor_type output_factor_type;

    //! CRF factor regularization type.
    typedef typename crf_factor::regularization_type crf_factor_reg_type;

    //! Parameters type.
    typedef pwl_crf_weights_parameters<crf_factor> parameters;

    // Protected data members
    //==========================================================================
  protected:

    parameters params;

    size_t learning_mode_;

    mutable boost::mt11213b rng;

    mutable boost::uniform_int<int> unif_int;

    //! Training data.
    const dataset<>& ds;

    //! Output variables.
    output_domain_type Yvars_;

    //! Mapping specifying which X to use for each factor over subsets of Y.
    crf_X_mapping<crf_factor> X_mapping_;

    /**
     * For saving edge score info for each edge considered.
     * Map: edge pair (ordered first < second)
     *       --> <edge part of score, y1 part of score, y2 part of score>
     * (Only retained when set in parameters.)
     */
    mutable std::map<std::pair<output_variable_type*,output_variable_type*>,vec>
      edge_score_info_;

    //! (For templated learning mode, or when saved b/c of parameter setting)
    //! map: vertex pair <y1,y2> --> regressor for edge part of score
    //! Vertex pairs are stored s.t. y1 < y2.
    mutable std::map<std::pair<output_variable_type*,output_variable_type*>,
                     crf_factor>
      edge_part_map_;

    //! (For templated learning mode, or when saved b/c of parameter setting)
    //! map: vertex pair <y1,y2> --> regressors for vertex parts of score
    //! Vertex pairs are stored s.t. y1 < y2.
    mutable std::map<std::pair<output_variable_type*,output_variable_type*>,
                     std::pair<crf_factor, crf_factor> >
      vertex_part_map_;

    // Temporary for avoiding reallocation.
    mutable output_factor_type tmp_fctr;

    // Protected methods
    //==========================================================================

    //! Compute regressor P(Yvars | X_Yvars)
    crf_factor compute_regressor(const output_domain_type& Y) const {
      if (params.crf_factor_cv) {
        return
          learn_crf_factor<crf_factor>::train_cv
          (params.cv_params, ds, Y, X_mapping_[Y],
           *(params.crf_factor_params_ptr), unif_int(rng));
      } else {
        return
          learn_crf_factor<crf_factor>::train
          (ds, Y, X_mapping_[Y], *(params.crf_factor_params_ptr),
           unif_int(rng));
      }
    }

    /**
     * Initialize stuff.
     * Pre-compute regressors if using templated learning mode.
     */
    void init() {
      assert(params.valid());
      rng.seed(params.random_seed);
      unif_int = boost::uniform_int<int>(0,std::numeric_limits<int>::max());
      assert(ds.size() > 0);
      assert(includes(ds.variables(), Yvars_));

      switch (learning_mode_) {
      case 0:
        break;
      case 1:
        {
          // Pre-compute necessary regressors for each edge.
          subset_iterator<output_domain_type> Yvars_end;
          for (subset_iterator<output_domain_type> Yvars_it(Yvars_, 2);
             Yvars_it != Yvars_end; ++Yvars_it) {
            const output_domain_type& twoYvars = *Yvars_it;
            typename output_domain_type::const_iterator
              twoYvars_it(twoYvars.begin());
            assert(twoYvars_it != twoYvars.end());
            output_variable_type* y1 = *twoYvars_it;
            ++twoYvars_it;
            assert(twoYvars_it != twoYvars.end());
            output_variable_type* y2 = *twoYvars_it;
            if (y1 > y2)
              std::swap(y1,y2);
            std::pair<output_variable_type*,output_variable_type*>
              y12pair(std::make_pair(y1,y2));
            switch (params.score_type) {
            case 0: // PWL
              edge_part_map_[y12pair] = compute_regressor(twoYvars);
              break;
            case 1: // DCI
              {
                crf_factor f1(compute_regressor(make_domain(y1)));
                crf_factor f2(compute_regressor(make_domain(y2)));
                vertex_part_map_[y12pair] = std::make_pair(f1,f2);
                edge_part_map_[y12pair] = compute_regressor(twoYvars);
              }
              break;
            case 2: // CMI
              edge_part_map_[y12pair] = compute_regressor(twoYvars);
              break;
            default:
              assert(false);
            }
          }
        }
        break;
      default:
        assert(false);
      }
    }

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Constructor for a functor for computing edge weights.
     *
     * This can be used for different learning modes:
     *  - 0: Regular models.
     *       This allows the functor to compute scores for
     *       edges and return regressors for P(Y_i,Y_j | X_{ij}).
     *       See, e.g., pwl_crf_learner.
     *  - 1: Templated models.
     *       This allows the functor to pre-compute regressors for all edges
     *       which can be used later to compute edge scores given X = x.
     *       See, e.g., templated_tree_crf.
     *
     * @param ds             Training dataset.
     * @param Yvars          Y variables which will be used by this functor.
     * @param X_mapping_     Map specifying which X vars to use for each factor.
     *                       Note this type supports automatic conversions from
     *                       various types of input mappings.
     * @param learning_mode  0 = regular; 1 = templated.
     */
    pwl_crf_weights
    (const dataset<>& ds, const output_domain_type& Yvars_,
     const crf_X_mapping<crf_factor>& X_mapping_, bool learning_mode_,
     parameters params = parameters())
      : params(params), learning_mode_(learning_mode_), ds(ds),
        Yvars_(Yvars_), X_mapping_(X_mapping_) {
      init();
    }

    //! Returns a map:
    //!    edge pair (ordered first < second)
    //!     --> <edge part of score, y1 part of score, y2 part of score>
    //! (You must set a parameter to retain this full mapping.)
    const std::map<std::pair<output_variable_type*,output_variable_type*>, vec>&
    edge_score_info() const {
      return edge_score_info_;
    }

    /**
     * Returns a map:
     *    edge pair (ordered first < second)
     *     --> regression function used for edge part of score
     * (You must set a parameter to retain this mapping.)
     */
    const std::map<std::pair<output_variable_type*,output_variable_type*>,
                   crf_factor>&
    edge_part_map() const {
      return edge_part_map_;
    }

    /**
     * Returns a map:
     *    edge pair (ordered first < second)
     *     --> regression functions used for vertex parts of score
     * (You must set a parameter to retain this mapping.)
     */
    const std::map<std::pair<output_variable_type*,output_variable_type*>,
                   std::pair<crf_factor,crf_factor> >&
    vertex_part_map() const {
      return vertex_part_map_;
    }

    //! Clears edge_score_info, edge_part_map, vertex_part_map
    //! to free up space.
    void clear_info_maps() {
      edge_score_info_.clear();
      edge_part_map_.clear();
      vertex_part_map_.clear();
    }

    //! Clears edge_part_map to free up space.
    void clear_edge_part_map() {
      edge_part_map_.clear();
    }

    //! Output arguments.
    const output_domain_type& Yvars() const {
      return Yvars_;
    }

    //! Returns the crf_X_mapping being used.
    const crf_X_mapping<crf_factor>& X_mapping() const {
      return X_mapping_;
    }

    // Methods for regular learning mode (for non-templated models).
    //==========================================================================

    /**
     * Piecewise Likelihood (PWL).
     * Returns the score for the factor Phi(Y1,Y2, X_{12}).
     *
     * @return <edge score, P(Y1,Y2 | X_{12})>
     */
    std::pair<double, crf_factor>
    pwl(output_variable_type* y1, output_variable_type* y2) const {
      assert(learning_mode_ == 0);
      vec edge_score(zeros<vec>(3));
      assert(y1 && y2);
      assert((Yvars_.count(y1) != 0) && (Yvars_.count(y2) != 0));
      if (y1 > y2)
        std::swap(y1,y2);
      output_domain_type Y(make_domain(y1,y2));
      std::pair<output_variable_type*, output_variable_type*>
        y12pair(std::make_pair(y1,y2));
      crf_factor f(compute_regressor(Y));
      edge_part_map_[y12pair] = f;
      double total_ds_weight(0);
      size_t i(0);
      foreach(const record_type& r, ds.records()) {
        tmp_fctr = f.condition(r);
        tmp_fctr.normalize();
        edge_score[0] += ds.weight(i) * tmp_fctr.logv(r);
        total_ds_weight += ds.weight(i);
        ++i;
      }
      assert(total_ds_weight > 0);
      if (params.retain_edge_score_info) {
        edge_score_info_[y12pair] = edge_score;
      }
      return std::make_pair(sum(edge_score) / total_ds_weight, f);
    }

    /**
     * Decomposable Conditional Influence (DCI).
     * Returns the score for the factor Phi(Y1,Y2, X_{12}).
     *
     * @return <edge score, P(Y1,Y2 | X_{12})>
     */
    std::pair<double, crf_factor>
    dci(output_variable_type* y1, output_variable_type* y2) const {
      assert(learning_mode_ == 0);
      vec edge_score(zeros<vec>(3));
      assert(y1 && y2);
      assert((Yvars_.count(y1) != 0) && (Yvars_.count(y2) != 0));
      if (y1 > y2)
        std::swap(y1,y2);
      output_domain_type Y(make_domain(y1,y2));
      output_domain_type Y1(make_domain(y1));
      output_domain_type Y2(make_domain(y2));
      std::pair<output_variable_type*, output_variable_type*>
        y12pair(std::make_pair(y1,y2));
      crf_factor f(compute_regressor(Y));
      crf_factor f1(compute_regressor(Y1));
      crf_factor f2(compute_regressor(Y2));
      edge_part_map_[y12pair] = f;
      if (params.retain_vertex_part_map)
        vertex_part_map_[y12pair] = std::make_pair(f1, f2);
      double total_ds_weight(0);
      size_t i(0);
      foreach(const record_type& r, ds.records()) {
        tmp_fctr = f.condition(r);
        tmp_fctr.normalize();
        edge_score[0] += ds.weight(i) * tmp_fctr.logv(r);
        tmp_fctr = f1.condition(r);
        tmp_fctr.normalize();
        edge_score[1] -= ds.weight(i) * tmp_fctr.logv(r);
        tmp_fctr = f2.condition(r);
        tmp_fctr.normalize();
        edge_score[2] -= ds.weight(i) * tmp_fctr.logv(r);
        total_ds_weight += ds.weight(i);
        ++i;
      }
      assert(total_ds_weight > 0);
      if (params.retain_edge_score_info) {
        edge_score_info_[y12pair] = edge_score;
      }
      return std::make_pair(sum(edge_score) / total_ds_weight, f);
    }

    /**
     * Conditional Mutual Information (CMI).
     * Returns the score for the factor Phi(Y1,Y2, X_{12}).
     *
     * @return <edge score, P(Y1,Y2 | X_{12})>
     */
    std::pair<double, crf_factor>
    cmi(output_variable_type* y1, output_variable_type* y2) const {
      assert(learning_mode_ == 0);
      vec edge_score(zeros<vec>(3));
      assert(y1 && y2);
      assert((Yvars_.count(y1) != 0) && (Yvars_.count(y2) != 0));
      if (y1 > y2)
        std::swap(y1,y2);
      output_domain_type Y(make_domain(y1,y2));
      output_domain_type Y1(make_domain(y1));
      output_domain_type Y2(make_domain(y2));
      std::pair<output_variable_type*, output_variable_type*>
        y12pair(std::make_pair(y1,y2));
      crf_factor f(compute_regressor(Y));
      edge_part_map_[y12pair] = f;
      double total_ds_weight(0);
      size_t i(0);
      foreach(const record_type& r, ds.records()) {
        tmp_fctr = f.condition(r);
        tmp_fctr.normalize();
        edge_score[0] += ds.weight(i) * tmp_fctr.logv(r);
        edge_score[1] -= ds.weight(i) * tmp_fctr.marginal(Y1).logv(r);
        edge_score[2] -= ds.weight(i) * tmp_fctr.marginal(Y2).logv(r);
        total_ds_weight += ds.weight(i);
        ++i;
      }
      assert(total_ds_weight > 0);
      if (params.retain_edge_score_info) {
        edge_score_info_[y12pair] = edge_score;
      }
      return std::make_pair(sum(edge_score) / total_ds_weight, f);
    }

    /**
     * Returns the score for the factor Phi(Y1,Y2, X_{12}),
     * using the scoring method selected in the parameters.
     *
     * @return <edge score, P(Y1,Y2 | X_{12})>
     */
    std::pair<double, crf_factor>
    operator()(output_variable_type* y1, output_variable_type* y2) const {
      switch(params.score_type) {
      case 0:
        return pwl(y1,y2);
        break;
      case 1:
        return dci(y1,y2);
        break;
      case 2:
        return cmi(y1,y2);
        break;
      default:
        assert(false);
        return std::make_pair(- std::numeric_limits<double>::infinity(),
                              crf_factor());
      }
    }

    // Methods for templated learning mode (for templated models).
    //==========================================================================

    /**
     * Returns the score for the factor Phi(Y1,Y2, X_{12}),
     * using the scoring method selected in the parameters,
     * given the evidence X = x.
     * To get the estimate of P(Y1,Y2 | X_{12}=x_{12}) used to compute the
     * score, call get_last_regressor() directly after calling this method.
     *
     * @param x  Evidence
     *           This can be type input_assignment_type or input_record_type.
     *
     * @tparam XType  Type holding values X=x.
     */
    template <typename XType>
    double
    operator()(output_variable_type* y1, output_variable_type* y2,
               const XType& x) const {
      assert(learning_mode_ == 1);
      assert(y1 && y2);
      if (y1 > y2)
        std::swap(y1,y2);
      std::pair<output_variable_type*,output_variable_type*>
        y12pair(std::make_pair(y1,y2));
      switch(params.score_type) {
      case 0:
        {
          const crf_factor& f = safe_get(edge_part_map_, y12pair);
          tmp_fctr = f.condition(x);
          tmp_fctr.normalize();
          double score = tmp_fctr.logv(x);
          if (params.retain_edge_score_info) {
            vec edge_score(zeros<vec>(3));
            edge_score[0] = score;
            edge_score_info_[y12pair] = edge_score;
          }
          return score;
        }
        break;
      case 1:
        {
          vec edge_score(zeros<vec>(3));
          const std::pair<crf_factor, crf_factor>&
            f2 = safe_get(vertex_part_map_, y12pair);
          tmp_fctr = f2.first.condition(x);
          tmp_fctr.normalize();
          edge_score[1] = -tmp_fctr.logv(x);
          tmp_fctr = f2.second.condition(x);
          tmp_fctr.normalize();
          edge_score[2] = -tmp_fctr.logv(x);
          const crf_factor& f = safe_get(edge_part_map_, y12pair);
          tmp_fctr = f.condition(x);
          tmp_fctr.normalize();
          edge_score[0] = tmp_fctr.logv(x);
          if (params.retain_edge_score_info) {
            edge_score_info_[y12pair] = edge_score;
          }
          return sum(edge_score);
        }
        break;
      case 2:
        {
          vec edge_score(zeros<vec>(3));
          const crf_factor& f = safe_get(edge_part_map_, y12pair);
          tmp_fctr = f.condition(x);
          tmp_fctr.normalize();
          edge_score[0] = tmp_fctr.logv(x);
          edge_score[1] = -tmp_fctr.marginal(make_domain(y1)).logv(x);
          edge_score[2] = -tmp_fctr.marginal(make_domain(y2)).logv(x);
          if (params.retain_edge_score_info) {
            edge_score_info_[y12pair] = edge_score;
          }
          return sum(edge_score);
        }
        break;
      default:
        assert(false);
        return -std::numeric_limits<double>::infinity();
      }
    }

    /**
     * Returns the estimate of P(Y1,Y2 | X_{12}=x_{12}) used in the
     * last edge score computed.
     * Note: The returned factor may be invalidated by another
     *       call to this class' methods.
     */
    const output_factor_type& get_last_regressor() const {
      return tmp_fctr;
    }

  }; // class pwl_crf_weights

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_PWL_CRF_WEIGHTS_HPP
