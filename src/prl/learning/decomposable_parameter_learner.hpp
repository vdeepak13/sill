#ifndef PRL_DECOMPOSABLE_PARAMETER_LEARNER_HPP
#define PRL_DECOMPOSABLE_PARAMETER_LEARNER_HPP

#include <boost/timer.hpp>

#include <prl/factor/concepts.hpp>
#include <prl/learning/crossval_methods.hpp>
#include <prl/learning/dataset/dataset_view.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/learn_factor.hpp>
#include <prl/math/free_functions.hpp>
#include <prl/math/statistics.hpp>
#include <prl/model/decomposable.hpp>

#include <prl/macros_def.hpp>

namespace prl {

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

    //! The type of potentials/factors.
    typedef F factor_type;

    //! The decomposable model type.
    typedef decomposable<factor_type> model_type;

    //! Learning options.
    typedef decomposable_parameter_learner_parameters parameters;

    //! Functor used for cross validation to choose lambda;
    //! computes the score for a single record.
    //! This score is meant to be minimized.
    struct cross_val_functor {

      const model_type& model_;

      size_t score_type;

      cross_val_functor(const model_type& model_, size_t score_type)
        : model_(model_), score_type(score_type) { }

      double operator()(const record& r) const {
        switch(score_type) {
        case 0: // log likelihood
          return - model_.log_likelihood(r);
        case 1: // per-label accuracy
          return - model_.per_label_accuracy(r);
        case 2: // all-or-nothing label accuracy
          return - model_.accuracy(r);
        case 3: // mean squared error
                // (\propto per-label accuracy for finite data)
          return model_.mean_squared_error(r);
        default:
          assert(false);
          return std::numeric_limits<double>::infinity();
        }
      }

    }; // struct cross_val_functor

    // Public methods
    // =========================================================================

    /**
     * Constructor.
     * This learns parameters for the given structure from data.
     */
    decomposable_parameter_learner
    (const typename model_type::jt_type& structure,
     const dataset& ds, const parameters& params = parameters())
      : params(params) {
      model_.clear();
      std::vector<factor_type> marginals;
      foreach(const typename decomposable<factor_type>::vertex& v,
              structure.vertices()) {
        marginals.push_back(learn_marginal<factor_type>
                            (structure.clique(v), ds, params.regularization));
      }
      model_.initialize(structure, marginals);
    }

    //! Return the learned model.
    const model_type& model() const {
      return model_;
    }

    /**
     * Choose regularization parameters via n-fold cross validation.
     *
     * @param reg_params  (Return value.) regularization settings tried
     * @param means       (Return value.) means[i] = avg score for reg_params[i]
     * @param stderrs     (Return value.) corresponding std errors of scores
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
     *
     * @todo Change this to sort lambdas and do them in decreasing order,
     *       using previous results to warm-start each optimization.
     */
    static
    double choose_lambda
    (std::vector<double>& reg_params, vec& means, vec& stderrs,
     const crossval_parameters<1>& cv_params,
     const typename model_type::jt_type& structure, const dataset& ds,
     const parameters& params, size_t score_type, unsigned random_seed) {
      assert(score_type < 4);
      choose_lambda_helper clh(structure, ds, score_type, params);
      std::vector<vec> lambdas;
      vec best_lambda =
        crossval_zoom<choose_lambda_helper, 1>
        (lambdas, means, stderrs, cv_params, clh, random_seed);
      reg_params.clear();
      foreach(const vec& v, lambdas) {
        assert(v.size() == 1);
        reg_params.push_back(v[0]);
      }
      assert(best_lambda.size() == 1);
      return best_lambda[0];
    }

    // Private types
    // =========================================================================
  private:

    //! Helper functor for choose_lambda().
    //! @see crossval_zoom
    struct choose_lambda_helper {

      const typename model_type::jt_type& structure;

      const dataset& ds;

      size_t score_type;

      const parameters& params_;

      choose_lambda_helper
      (const typename model_type::jt_type& structure, const dataset& ds,
       size_t score_type, const parameters& params_)
        : structure(structure), ds(ds), score_type(score_type),
          params_(params_) { }

      //! Run CV on the given lambdas.
      vec operator()(vec& means, vec& stderrs, const std::vector<vec>& lambdas,
                     size_t n_folds, unsigned random_seed) const {
        assert(lambdas.size() > 0);
        assert(n_folds > 0 && n_folds <= ds.size());
        for (size_t j(0); j < lambdas.size(); ++j)
          assert(lambdas[j].size() == 1);
        means.resize(lambdas.size());
        means.zeros_memset();
        stderrs.resize(lambdas.size());
        stderrs.zeros_memset();

        boost::mt11213b rng(random_seed);
        dataset_view permuted_view(ds);
        permuted_view.set_record_indices(randperm(ds.size(), rng));
        parameters fold_params(params_);
        boost::shared_ptr<dataset_view>
          fold_train_ptr(new dataset_view(permuted_view));
        dataset_view fold_test(permuted_view);
        fold_train_ptr->save_record_view();
        fold_test.save_record_view();
        for (size_t fold(0); fold < n_folds; ++fold) {
          if (fold != 0) {
            fold_train_ptr->restore_record_view();
            fold_test.restore_record_view();
          }
          fold_train_ptr->set_cross_validation_fold(fold, n_folds, false);
          fold_test.set_cross_validation_fold(fold, n_folds, true);
          // Make a hard copy of the training set for efficiency.
          vector_dataset tmp_train_ds(fold_train_ptr->datasource_info(),
                                      fold_train_ptr->size());
          foreach(const record& r, fold_train_ptr->records())
            tmp_train_ds.insert(r);
          for (size_t k(0); k < lambdas.size(); ++k) {
            fold_params.regularization = lambdas[k][0];
            fold_params.random_seed =
              boost::uniform_int<int>(0, std::numeric_limits<int>::max())(rng);
            try {
              boost::timer tmptimer;
              decomposable_parameter_learner dpl(structure, tmp_train_ds,
                                                 fold_params);
              if (params_.debug > 0)
                std::cerr << "Doing CV (fold " << fold
                          <<"): decomposable parameter learning time: "
                          << tmptimer.elapsed() << " seconds." << std::endl;
              double tmpval(fold_test.expected_value
                            (cross_val_functor
                             (dpl.model(), score_type)).first);
              if (is_finite(means[k])) {
                means[k] += tmpval;
                stderrs[k] += tmpval * tmpval;
              }
            } catch(normalization_error exc) {
              // Assume that the regularization is too weak.
              means[k] = std::numeric_limits<double>::infinity();
              stderrs[k] = std::numeric_limits<double>::infinity();
            }
          }
        }
        for (size_t k(0); k < lambdas.size(); ++k) {
          if (is_finite(means[k])) {
            means[k] /= n_folds;
            stderrs[k] /= n_folds;
            stderrs[k] = sqrt((stderrs[k] - means[k] * means[k]) / n_folds);
          }
        }
        size_t min_i(min_index(means, rng));
        if (!is_finite(means[min_i])) {
          std::cerr << "lambdas:\n";
          foreach(const vec& lambda, lambdas)
            std::cerr << "\t " << lambda << "\n";
          std::cerr << "\n"
                    << "means: " << means << "\n"
                    << "stderrs: " << stderrs << "\n"
                    << std::endl;
          throw std::runtime_error("decomposable_parameter_learner::choose_lambda_cv() ran into numerical problems for all possible lambda settings.");
        }
        if (params_.debug > 0) {
          std::cerr << "decomposable_parameter_learner::choose_lambda_cv()\n"
                    << "   scores:  " << means << "\n"
                    << "   stderrs: " << stderrs << "\n"
                    << "  Chosen parameters: " << lambdas[min_i]
                    << std::endl;
        }
        return lambdas[min_i];

      } // operator()

    }; // struct choose_lambda_helper

    // Private data members
    // =========================================================================

    parameters params;

    //! The learned model.
    model_type model_;

  }; // decomposable_parameter_learner

};  // namespace prl

#include <prl/macros_undef.hpp>

#endif // PRL_DECOMPOSABLE_PARAMETER_LEARNER_HPP
