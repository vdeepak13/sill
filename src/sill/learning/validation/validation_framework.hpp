#ifndef SILL_VALIDATION_FRAMEWORK_HPP
#define SILL_VALIDATION_FRAMEWORK_HPP

#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/validation/validation_functor.hpp>
#include <sill/math/permutations.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for automating validation and cross-validation.
   *
   * This can be used to:
   *  - Choose parameters via validation or cross-validation.
   *  - Run a test multiple times and get statistics about the results.
   *
   * "Validation" refers to testing on a held-out set of data.
   * "Cross-validation" refers to K-fold cross-validation.
   *
   * The constructors run the tests, after which the following results are
   * stored in this class:
   *  - For each parameter value tested (only 1 if not choosing parameters)
   *     - For each result type
   *        - For each test run
   *           - Result value
   *
   * TO DO: ADD TIMING
   */
  class validation_framework {

    // Public types
    // =========================================================================
  public:

    //! Parameters for the validation_framework.
    struct parameters {
    }; // struct parameters

    //! Type of validation being done.
    enum build_type_enum { CHOOSE_PARAM_CV,
                           CHOOSE_PARAM_VALIDATION,
                           TEST_CV,
                           TEST_VALIDATION };

    // Constructors
    // =========================================================================
  public:

    /**
     * Constructor: Choose parameters via K-fold cross validation.
     *
     * @tparam N  Number of elements in parameter vector being chosen.
     *
     * @param ds             Dataset.
     * @param train_functor  Model training functor.
     */
    template <size_t N>
    validation_framework(const dataset& ds,
                         const crossval_parameters<N>& cv_params,
                         model_training_functor& train_func,
                         unsigned random_seed = time(NULL));

    /**
     * Constructor: Choose parameters using a separate validation dataset.
     *
     * @tparam N  Number of elements in parameter vector being chosen.
     *
     * @param train_ds       Training data.
     * @param test_ds        Separate validation dataset.
     * @param train_functor  Model training functor.
     */
    template <size_t N>
    validation_framework(const dataset& train_ds,
                         const dataset& test_ds,
                         const crossval_parameters<N>& cv_params,
                         model_training_functor& train_func,
                         unsigned random_seed = time(NULL));

    /**
     * Constructor: Run a test multiple times (using K-fold cross validation),
     *              and get statistics about the results.
     *
     * @param ds             Dataset.
     * @param train_functor  Model training functor.
     */
    validation_framework(const dataset& ds,
                         model_training_functor& train_func,
                         unsigned random_seed = time(NULL));

    /**
     * Constructor: Run a test multiple times (using different random seeds)
     *              on the same training and validation sets,
     *              and get statistics about the results.
     *
     * @param train_ds       Training data.
     * @param test_ds        Separate validation dataset.
     * @param train_functor  Model training functor.
     */
    validation_framework(const dataset& train_ds,
                         const dataset& test_ds,
                         model_training_functor& train_func,
                         unsigned random_seed = time(NULL));

    // Methods for getting results
    // =========================================================================

    //! Type of validation which was done
    //! (based on which constructor was called).
    build_type_enum build_type() const {
      return build_type_;
    }

    /**
     * Get a result statistic
     * RIGHT HERE NOW
     * Use build_type_ to check to make sure people are asking for the correct
     * results.
     */

    // Protected data
    // =========================================================================
  protected:

    build_type_enum build_type_;

    boost::mt11213b rng;

    boost::uniform_int<int> unif_int;

    //! List of all model parameters tested.
    std::vector<vec> lambdas_;

    //! results_[i] = list of results for each test run for lambdas_[i]
    std::vector<vec> results_;

    //! all_results_[result type][i]
    //!   = list of results for each test run for lambdas_[i]
    std::map<std::string, std::vector<vec> > all_results_;

    /**
     * When only running one test per lambda setting,
     *   single_results_[i] = result for lambdas_[i]
     * When running multiple tests per lambda setting,
     *   single_results_[i] = mean (or median, etc.) results for lambdas_[i]
     */
    vec single_results_;

    /**
     * When only running one test per lambda setting,
     *   single_all_results_[result type][i] = result for lambdas_[i]
     * When running multiple tests per lambda setting,
     *   single_all_results_[result type][i] =
     *         mean (or median, etc.) results for lambdas_[i]
     */
    std::map<std::string, vec> single_all_results_;

    // Protected methods
    // =========================================================================

    //! Update lambdas_, single_results_, single_all_results_.
    void
    add_single_results(size_t zoom_i, const std::vector<vec>& lambdas_zoom,
                       const validation_functor& val_func) {
      if (zoom_i == 0) {
        lambdas_ = lambdas_zoom;
        single_results_ = val_func.results;
        single_all_results_ = val_func.all_results;
      } else {
        size_t oldsize(lambdas_.size());
        lambdas_.resize(oldsize + lambdas_zoom.size());
        single_results_.resize(oldsize + lambdas_zoom.size(), true);
        assert(lambdas_zoom.size() == val_func.results.size());
        for (size_t j(0); j < lambdas_zoom.size(); ++j) {
          lambdas_[oldsize + j] = lambdas_zoom[j];
          single_results_[oldsize + j] = val_func.results[j];
        }
        typedef std::pair<std::string, vec> string_vec_pair;
        foreach(const string_vec_pair& svp, val_func.all_results) {
          assert(lambdas_zoom.size() == svp.second.size());
          single_all_results_[svp.first].resize(oldsize + lambdas_zoom.size(),
                                                true);
          for (size_t j(0); j < lambdas_zoom.size(); ++j) {
            single_all_results_[svp.first][oldsize + j] = svp.second[j];
          }
        }
      }
    } // add_single_results

    //! Update lambdas_, results_, all_results_.
    void
    add_results(size_t zoom_i, const std::vector<vec>& lambdas_zoom,
                const validation_functor& val_func,
                size_t run_i, size_t num_runs) {
      typedef std::pair<std::string, vec> string_vec_pair;
      assert(lambdas_zoom.size() == val_func.results.size());
      if (run_i == 0) {
        if (zoom_i == 0) {
          lambdas_.clear();
          results_.clear();
          all_results_.clear();
          /*
          foreach(const string_vec_pair& svp, val_func.all_results) {
            all_results_[svp.first] =
              std::vector<vec>(lambdas_zoom.size(), vec(num_runs, 0));
          }
          */
        } else {
          assert(val_func.all_results.size() == all_results_.size());
        }
        size_t oldsize(lambdas_.size());
        lambdas_.resize(oldsize + lambdas_zoom.size());
        results_.resize(oldsize + lambdas_zoom.size());
        for (size_t i = 0; i < lambdas_zoom.size(); ++i) {
          lambdas_[oldsize + i] = lambdas_zoom[i];
          results_[oldsize + i].resize(num_runs);
          results_[oldsize + i].zeros();
          results_[oldsize + i][run_i] = val_func.results[i];
        }
        foreach(const string_vec_pair& svp, val_func.all_results) {
          assert(lambdas_zoom.size() == svp.second.size());
          all_results_[svp.first].resize(oldsize + lambdas_zoom.size());
          for (size_t i = 0; i < lambdas_zoom.size(); ++i) {
            all_results_[svp.first][oldsize + i].resize(num_runs);
            all_results_[svp.first][oldsize + i].zeros();
            all_results_[svp.first][oldsize + i][run_i] = svp.second[i];
          }
        }
      } else {
        // Not first run in zoom level.
        assert(val_func.all_results.size() == all_results_.size());
        if (lambdas_zoom.size() > lambdas_.size()) {
          throw std::runtime_error
            ("validation_framework::add_results had internal failure!");
        }
        for (size_t i = lambdas_.size() - lambdas_zoom.size();
             i < lambdas_.size(); ++i) {
          results_[i][run_i] = val_func.results[i];
        }
        foreach(const string_vec_pair& svp, val_func.all_results) {
          assert(lambdas_zoom.size() == svp.second.size());
          for (size_t i = lambdas_.size() - lambdas_zoom.size();
               i < lambdas_.size(); ++i) {
            all_results_[svp.first][i][run_i] = svp.second[i];
          }
        }
      }
    } // add_results

    //! Combine results_ into single_results_
    //! and all_results_ into single_all_results_.
    template <size_t N>
    void combine_results(const crossval_parameters<N>& cv_params) {
      single_results_.resize(results_.size());
      for (size_t i = 0; i < results_.size(); ++i) {
        single_results_[i] =
          generalized_mean(results_[i], cv_params.run_combo_type);
      }
      single_all_results_.clear();
      foreach(const std::string& s, keys(all_results_)) {
        single_all_results_[s].resize(results_.size());
        for (size_t i = 0; i < results_.size(); ++i) {
          single_all_results_[s][i] = 
            generalized_mean(all_results_[s][i], cv_params.run_combo_type);
        }
      }
    } // combine_results

  }; // class validation_framework

  // =========================================================================
  // Implementations of templated methods in validation_framework
  // =========================================================================

    // Choose parameters via CV.
    template <size_t N>
    validation_framework::
    validation_framework(const dataset& ds,
                         const crossval_parameters<N>& cv_params,
                         model_training_functor& train_func,
                         unsigned random_seed)
      : build_type_(CHOOSE_PARAM_CV), rng(random_seed),
        unif_int(0, std::numeric_limits<int>::max()) {

      if (!cv_params.valid()) {
        throw std::invalid_argument
          ("validation_framework constructed with invalid cv_params.");
      }

      // These hold values for each round of zooming:
      std::vector<vec> lambdas_zoom =
        create_parameter_grid(cv_params.minvals, cv_params.maxvals,
                              cv_params.nvals, cv_params.log_scale);

      validation_functor val_func;

      // Set up dataset views.
      dataset_view permuted_view(ds);
      permuted_view.set_record_indices(randperm(ds.size(), rng));
      dataset_view fold_train(permuted_view);
      dataset_view fold_test(permuted_view);
      fold_train.save_record_view();
      fold_test.save_record_view();

      // For each level of lambda zooming,
      for (size_t zoom_i(0); zoom_i <= cv_params.zoom; ++zoom_i) {
        // For each fold of CV,
        for (size_t fold(0); fold < cv_params.nfolds; ++fold) {
          if (fold != 0) {
            fold_train.restore_record_view();
            fold_test.restore_record_view();
          }
          fold_train.set_cross_validation_fold(fold, cv_params.nfolds, false);
          fold_test.set_cross_validation_fold(fold, cv_params.nfolds, true);
          // Test lambdas_zoom.
          val_func.test(lambdas_zoom, fold_train, fold_test, unif_int(rng),
                        train_func);
          // Add to lambdas_, results_, all_results_.
          add_results(zoom_i, lambdas_zoom, val_func, fold, cv_params.nfolds);
        }

        // Combine results_ into single_results_
        // and all_results_ into single_all_results_.
        combine_results(cv_params);

        // Prepare next lambdas_zoom.
        if (zoom_i + 1 <= cv_params.zoom) {
          size_t best_i = min_index(single_results_);
          lambdas_zoom =
            zoom_parameter_grid(lambdas_, lambdas_[best_i], cv_params.nvals,
                                cv_params.log_scale);
        }
      } // loop over zooms
    } // constructor (choose parameters via CV)

    // Choose parameters via validation set.
    template <size_t N>
    validation_framework::
    validation_framework(const dataset& train_ds,
                         const dataset& test_ds,
                         const crossval_parameters<N>& cv_params,
                         model_training_functor& train_func,
                         unsigned random_seed)
      : build_type_(CHOOSE_PARAM_VALIDATION), rng(random_seed),
        unif_int(0, std::numeric_limits<int>::max()) {

      if (!cv_params.valid()) {
        throw std::invalid_argument
          ("validation_framework constructed with invalid cv_params.");
      }

      // These hold values for each round of zooming:
      std::vector<vec> lambdas_zoom =
        create_parameter_grid(cv_params.minvals, cv_params.maxvals,
                              cv_params.nvals, cv_params.log_scale);

      validation_functor val_func;

      for (size_t zoom_i(0); zoom_i <= cv_params.zoom; ++zoom_i) {
        // Test lambdas_zoom.
        val_func.test(lambdas_zoom, train_ds, test_ds, unif_int(rng),
                      train_func);
        // Add to lambdas_, single_results_, single_all_results_.
        add_single_results(zoom_i, lambdas_zoom, val_func);
        // Prepare next lambdas_zoom.
        if (zoom_i + 1 <= cv_params.zoom) {
          size_t best_i = min_index(single_results_);
          lambdas_zoom =
            zoom_parameter_grid(lambdas_, lambdas_[best_i], cv_params.nvals,
                                cv_params.log_scale);
        }
      } // loop over zooms
    } // constructor (choose parameters via validation set)

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_VALIDATION_FRAMEWORK_HPP
