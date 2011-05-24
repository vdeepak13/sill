#ifndef SILL_DECOMPOSABLE_PARAMETER_LEARNER_HPP
#define SILL_DECOMPOSABLE_PARAMETER_LEARNER_HPP

#include <boost/timer.hpp>

#include <sill/factor/concepts.hpp>
#include <sill/learning/crossval_methods.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/learn_factor.hpp>
#include <sill/learning/validation/validation_framework.hpp>
#include <sill/math/permutations.hpp>
#include <sill/math/statistics.hpp>
#include <sill/model/decomposable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  template <typename F> class decomposable_parameter_learner_val_functor;

  /**
   * Parameters for decomposable_parameter_learner.
   *
   * This allows easy parsing of command-line options via Boost Program Options.
   *
   * Usage: Create your own Options Description desc.
   *        Call this struct's add_options() method with desc to add synthetic
   *        model options to desc.
   *        Parse the command line using the modified options description.
   *        Pass this struct (which now holds the parsed options) to
   *        decomposable_parameter_learner.
   */
  struct decomposable_parameter_learner_parameters {

    /**
     * Regularization / smoothing passed to methods for learning marginals.
     * This must be >= 0.
     * (default = 0)
     */
    double regularization;

    /**
     * Random seed.
     *  (default = time)
     */
    unsigned random_seed;

    /**
     * Print debugging info.
     *  - 0: none (default)
     *  - 1: some
     *  - higher: reverts to highest debugging mode
     */
    size_t debug;

    // Methods
    //==========================================================================

    decomposable_parameter_learner_parameters()
      : regularization(0), random_seed(time(NULL)), debug(0) { }

    /**
     * @param verbose  If true, print warnings to STDERR about invalid options.
     *                 (default = true)
     *
     * @return true iff the parameters are valid
     */
    bool valid(bool verbose = true) const;

  }; // struct decomposable_parameter_learner_parameters

  /**
   * This is a class which represents algorithms for learning a decomposable
   * model from data.
   * This treats models as a collection of factors with these requirements:
   *  - Instantiating the factors should produce a LOW TREE-WIDTH decomposable
   *    model.
   *  - The factor type supports learning marginal distributions from data.
   * 
   * @tparam F  factor/potential type which implements the
   *            LearnableDistributionFactor concept
   *
   * @see decomposable
   * \ingroup learning_param
   */
  template <typename F>
  class decomposable_parameter_learner {

//    concept_assert((LearnableDistributionFactor<F>));

    // Public types
    // =========================================================================
  public:

    typedef dense_linear_algebra<> la_type;

    typedef record<la_type> record_type;

    //! The type of potentials/factors.
    typedef F factor_type;

    //! The decomposable model type.
    typedef decomposable<factor_type> model_type;

    //! Learning options.
    typedef decomposable_parameter_learner_parameters parameters;

    // Public methods
    // =========================================================================

    /**
     * Constructor.
     * This learns parameters for the given structure from data.
     */
    decomposable_parameter_learner
    (const typename model_type::jt_type& structure,
     const dataset<la_type>& ds, const parameters& params = parameters())
      : params(params) {
      model_.clear();
      std::vector<factor_type> marginals;
      foreach(const typename decomposable<factor_type>::vertex& v,
              structure.vertices()) {
        marginals.push_back(learn_factor<factor_type>::learn_marginal
                            (structure.clique(v), ds, params.regularization));
      }
      model_.initialize(structure, marginals);
    }

    //! Return the learned model.
    const model_type& model() const {
      return model_;
    }

    /**
     * Choose regularization parameters via k-fold cross validation.
     *
     * @param cv_params   Parameters specifying how to do cross validation.
     * @param structure   Decomposable structure for which to learn parameters.
     * @param ds          Training data.
     * @param params      Parameters for this class.
     * @param score_type  0: log likelihood, 1: per-label accuracy,
     *                    2: all-or-nothing label accuracy,
     *                    3: mean squared error
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  chosen regularization parameters
     */
    static
    double choose_lambda
    (const crossval_parameters& cv_params,
     const typename model_type::jt_type& structure,
     const dataset<la_type>& train_ds,
     const parameters& params, size_t score_type, unsigned random_seed) {
      assert(score_type == 0); // others not yet implemented
      decomposable_parameter_learner_val_functor<F>
        dpl_val_func(structure, params);
      validation_framework<la_type>
        val_frame(train_ds, cv_params, dpl_val_func, random_seed);
      vec best_lambdas(val_frame.best_lambdas());
      assert(best_lambdas.size() == 1);
      return best_lambdas[0];
    }

    /**
     * Choose regularization parameters via separate training and validation
     * datasets.
     *
     * @param cv_params   Parameters specifying how to do cross validation.
     * @param structure   Decomposable structure for which to learn parameters.
     * @param train_ds    Training data.
     * @param val_ds      Validation data.
     * @param params      Parameters for this class.
     * @param score_type  0: log likelihood, 1: per-label accuracy,
     *                    2: all-or-nothing label accuracy,
     *                    3: mean squared error
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  chosen regularization parameters
     */
    static
    double choose_lambda
    (const crossval_parameters& cv_params,
     const typename model_type::jt_type& structure,
     const dataset<la_type>& train_ds, const dataset<la_type>& val_ds,
     const parameters& params, size_t score_type, unsigned random_seed) {
      assert(score_type == 0); // others not yet implemented
      decomposable_parameter_learner_val_functor<F>
        dpl_val_func(structure, params);
      validation_framework<la_type>
        val_frame(train_ds, val_ds, cv_params, dpl_val_func, random_seed);
      vec best_lambdas(val_frame.best_lambdas());
      assert(best_lambdas.size() == 1);
      return best_lambdas[0];
    }

    // Private data members
    // =========================================================================
  private:

    parameters params;

    //! The learned model.
    model_type model_;

  }; // decomposable_parameter_learner

  /**
   * Model validation functor for decomposable models.
   *
   * @tparam F  Factor type.
   */
  template <typename F>
  class decomposable_parameter_learner_val_functor
    : public model_validation_functor<> {

    // Public types
    // =========================================================================
  public:

    typedef model_validation_functor<> base;

    //! Type of factor
    typedef F factor_type;

    //! Type of graph.
    typedef typename decomposable<F>::jt_type jt_type;

    // Constructors and destructors
    // =========================================================================

    /**
     * Constructor which uses a jt_type.
     * WARNING: This does not work with templated factors; use the below
     *          constructor instead.
     */
    decomposable_parameter_learner_val_functor
    (const jt_type& structure,
     const decomposable_parameter_learner_parameters& dpl_params)
      : structure(structure), dpl_params(dpl_params) {
      this->use_weights = false;
    }

    /**
     * Constructor which uses a decomposable model.
     *
     * @param use_weights   If true, then use the given model's weights
     *                      (parameters) to initialize learning.
     */
    decomposable_parameter_learner_val_functor
    (const decomposable<F>& model, bool use_weights,
     const decomposable_parameter_learner_parameters& dpl_params)
      : structure(model.get_junction_tree()), model(model),
        dpl_params(dpl_params) {
      this->use_weights = use_weights;
    }

    // Protected data
    // =========================================================================
  protected:

    const jt_type& structure;

    decomposable<F> model;

    decomposable_parameter_learner_parameters dpl_params;

    // Protected methods
    // =========================================================================

    void train_model(const dataset<>& ds, unsigned random_seed) {
      dpl_params.random_seed = random_seed;
      if (model.num_arguments() != 0) {
//        decomposable_parameter_learner<F>
//          dpl(model, !use_weights, ds, dpl_params);

        decomposable_parameter_learner<F> // warm-starts not supported yet
          dpl(structure, ds, dpl_params);
        model = dpl.model();
      } else {
        std::cerr << "WARNING: This version of decomposable_parameter_learner_val_functor does not work with templated factors." << std::endl; // TO DO: Figure out a way to resolve this issue.
        decomposable_parameter_learner<F> dpl(structure, ds, dpl_params);
        model = dpl.model();
      }
    }

    void train_model(const dataset<>& ds, const vector_type& validation_params,
                     unsigned random_seed) {
      assert(validation_params.size() == 1);
      dpl_params.regularization = validation_params[0];
      train_model(ds, random_seed);
    }

    double add_results(const dataset<>& ds, const std::string& prefix) {
      double ll = model.expected_log_likelihood(ds);
      result_map_[prefix + "log likelihood"] = ll;
      result_map_[prefix + "per-label accuracy"] =
        model.expected_per_label_accuracy(ds);
      return ll;
    }

  }; // class decomposable_parameter_learner_val_functor

};  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_DECOMPOSABLE_PARAMETER_LEARNER_HPP
