
#ifndef SILL_PWL_CRF_PARAMETER_LEARNER_HPP
#define SILL_PWL_CRF_PARAMETER_LEARNER_HPP

#include <set>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/factor/log_reg_crf_factor.hpp>
#include <sill/iterator/subset_iterator.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/free_functions.hpp>
#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for learning the parameters of a CRF P(Y|X) using piecewise
   * likelihood.
   *
   * About piecewise likelihood:
   *  - It is based on a bound for the partition function, and
   *    it allows you to get a decomposable score which permits you to
   *    learn parameters for a CRF independently for each factor.
   *  - See "Piecewise training of undirected models"
   *    by C Sutton, A McCallum (2005).
   *
   * @tparam FactorType  type of factor which fits the CRFfactor concept
   *
   * \author Joseph Bradley
   * \ingroup learning_param
   * @todo This does not support tied parameters.
   */
  template <typename FactorType>
  class pwl_crf_parameter_learner {

    concept_assert((sill::LearnableCRFfactor<FactorType>));

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

    //! Graph type
    typedef typename crf_model_type::base graph_type;

    struct parameters {

    private:
      typedef typename crf_factor::regularization_type factor_reg_type;

    public:

      /**
       * CRF factor parameters specifying how the crf_factor type
       * calculates the conditional probabilities in the score.
       * Note that this contains any regularization parameters as well
       * but is overriden if crf_factor_choose_lambda is selected.
       *  (default = factor parameter defaults (if they exist!))
       */
      boost::shared_ptr<typename crf_factor::parameters> crf_factor_params_ptr;

      //! If true, do cross-validation to choose the regularization parameters
      //! for pairwise factors, using the below parameters n_folds, minvals,
      //! maxvals, nlambdas, and zoom.
      //!  (default = false)
      bool crf_factor_cv;

      //! Parameters specifying how to do cross validation.
      crossval_parameters<factor_reg_type::nlambdas> cv_params;

      /**
       * Used to make the algorithm deterministic
       *    (default = time)
       */
      unsigned random_seed;

      //! Debugging modes:
      //!  - 0: no debugging (default)
      //!  - 1: print progress through functions
      //!  - 2: print factor scores as they are computed
      //!  - above: same as highest debugging mode
      size_t DEBUG;

      parameters()
        : crf_factor_cv(false), random_seed(time(NULL)), DEBUG(0) { }

      bool valid() const {
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

    }; // class parameters

    // Protected data members
    //==========================================================================
  protected:

    parameters params;

    //! Learned CRF factor graph.
    crf_model<crf_factor> model_;

    //! Sum of piecewise likelihoods from all factors in the model.
    double total_score_;

    // Protected methods
    //==========================================================================

    //! Returns the score for the factor Phi(Yvars, X_Yvars),
    //! plus the trained factor.
    std::pair<double, crf_factor*>
    factor_score(boost::shared_ptr<dataset> ds_ptr,
                 const output_domain_type& Yvars,
                 copy_ptr<input_domain_type> Xvars_ptr,
                 boost::mt11213b& rng) const {
      crf_factor* r_ptr = NULL;
      if (params.crf_factor_cv) {
        std::vector<typename crf_factor::regularization_type> reg_params;
        vec means, stderrs;
        r_ptr =
          learn_crf_factor_cv<crf_factor>
          (reg_params, means, stderrs, params.cv_params,
           ds_ptr, Yvars, Xvars_ptr, *(params.crf_factor_params_ptr),
           boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
      } else {
        r_ptr =
          learn_crf_factor<crf_factor>
          (ds_ptr, Yvars, Xvars_ptr, *(params.crf_factor_params_ptr),
           boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
      }
      return std::make_pair(r_ptr->log_expected_value(*ds_ptr), r_ptr);
    } // factor_score()

    void build(boost::shared_ptr<dataset> ds_ptr, const graph_type& graph) {
      if (!params.crf_factor_params_ptr)
        params.crf_factor_params_ptr.reset
          (new typename crf_factor::parameters());
      assert(params.valid());
      boost::mt11213b rng(params.random_seed);

      if (params.DEBUG > 0) {
        std::cerr << "pwl_crf_parameter_learner::build():"
                  << " computing all factors..."
                  << std::endl;
      }
      foreach(const typename graph_type::vertex& v, graph.factor_vertices()) {
        std::pair<double, crf_factor*>
          score_f(factor_score(ds_ptr, graph.output_arguments(v),
                               graph.input_arguments_ptr(v), rng));
        if (params.DEBUG > 1)
          std::cerr << "  Computed factor; PWL = " << score_f.first
                    << std::endl;
        model_.add_factor(*(score_f.second));
        total_score_ += score_f.first;
        delete(score_f.second);
      }
    } // build()

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Constructor for a learner for a model for P(Y | X).
     * This is given a CRF graph and parametrizes the CRF model.
     *
     * @param ds_ptr        Training dataset.
     * @param graph         CRF graph.
     * @param parameters    algorithm parameters
     */
    pwl_crf_parameter_learner
    (boost::shared_ptr<dataset> ds_ptr, const graph_type& graph,
     parameters params = parameters())
      : params(params), total_score_(0) {
      build(ds_ptr, graph);
    }

    // Getters and helper methods
    //==========================================================================

    //! Returns the current model, parametrized using piecewise likelihood.
    const crf_model<crf_factor>& model() const {
      return model_;
    }

    //! Returns the sum of scores from all factors added to the model
    //! (not counting single-variable factors).
    double total_score() const {
      return total_score_;
    }

  }; // class pwl_crf_parameter_learner

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_PWL_CRF_PARAMETER_LEARNER_HPP
