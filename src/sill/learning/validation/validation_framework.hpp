#ifndef SILL_VALIDATION_FRAMEWORK_HPP
#define SILL_VALIDATION_FRAMEWORK_HPP

#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/validation/param_list_validation_functor.hpp>
#include <sill/math/permutations.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for automating validation and cross-validation.
   *
   * This can be used to:
   *  - Choose parameters via validation or cross-validation.
   *     - When choosing, this class assumes that the MAXIMUM result/score is
   *       the best.
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
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class validation_framework {

    // Public types
    // =========================================================================
  public:

    typedef LA la_type;

    typedef typename la_type::value_type value_type;

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
     *  (CHOOSE_PARAM_CV)
     *
     * @param ds             Dataset.
     * @param mv_func        Model validation functor.
     */
    validation_framework(const dataset<la_type>& ds,
                         const crossval_parameters& cv_params,
                         model_validation_functor<la_type>& mv_func,
                         unsigned random_seed = time(NULL));

    /**
     * Constructor: Choose parameters using a separate validation dataset.
     *  (CHOOSE_PARAM_VALIDATION)
     *
     * @param train_ds       Training data.
     * @param test_ds        Separate validation dataset.
     * @param mv_func        Model validation functor.
     */
    validation_framework(const dataset<la_type>& train_ds,
                         const dataset<la_type>& test_ds,
                         const crossval_parameters& cv_params,
                         model_validation_functor<la_type>& mv_func,
                         unsigned random_seed = time(NULL));

    /**
     * Constructor: Run a test multiple times (using K-fold cross validation),
     *              and get statistics about the results.
     *  (TEST_CV)
     *
     * This uses mean/stderr by default, but you can get all of the results
     * and compute median/MAD or the like yourself.
     *
     * @param ds             Dataset.
     * @param mv_func        Model validation functor.
     * @param k              Number of folds of CV to run.
     */
    validation_framework(const dataset<la_type>& ds,
                         model_validation_functor<la_type>& mv_func,
                         size_t k,
                         unsigned random_seed = time(NULL));

    /**
     * Constructor: Run a test multiple times (using different random seeds)
     *              on the same training and validation sets,
     *              and get statistics about the results.
     *  (TEST_VALIDATION)
     *
     * @param train_ds       Training data.
     * @param test_ds        Separate validation dataset.
     * @param mv_func        Model validation functor.
     */
    validation_framework(const dataset<la_type>& train_ds,
                         const dataset<la_type>& test_ds,
                         model_validation_functor<la_type>& mv_func,
                         unsigned random_seed = time(NULL));

    // Methods for getting results
    // =========================================================================

    //! Type of validation which was done
    //! (based on which constructor was called).
    build_type_enum build_type() const {
      return build_type_;
    }

    /**
     * Get the best lambda value.
     * TO DO: Use build_type_ to check to make sure people are asking for
     *        the correct results.
     */
    vec best_lambdas() const {
      if (lambdas_.size() == 0)
        return vec();
      assert(single_results_.size() == lambdas_.size());
      size_t best_i = max_index(single_results_);
      return lambdas_[best_i];
    }

    //! List of all model parameters tested.
    const std::vector<vec>& lambdas() const { return lambdas_; }

    //! Returns means for main result/score type.
    const vec& means() const { return single_results_; }

    //! Returns standard errors for main result/score type.
    vec stderrs() const {
      vec v(zeros<vec>(single_results_.size()));
      assert(single_results_.size() == results_.size());
      for (size_t i = 0; i < results_.size(); ++i) {
        v[i] = generalized_deviation(results_[i], run_combo_type);
      }
      return v;
    }

    //! Returns map: result_type --> means for that result.
    const std::map<std::string, vec>& all_means() const {
      return single_all_results_;
    }

    //! Returns map: result_type --> standard errors for that result.
    std::map<std::string, vec> all_stderrs() const {
      typedef std::pair<std::string, std::vector<vec> > string_vecvec_pair;
      std::map<std::string, vec> allv;
      foreach(const string_vecvec_pair& svvp, all_results_) {
        vec v(zeros<vec>(svvp.second.size()));
        assert(single_all_results_.count(svvp.first) &&
               safe_get(single_all_results_, svvp.first).size() == svvp.second.size());
        for (size_t i = 0; i < svvp.second.size(); ++i) {
          v[i] = generalized_deviation(svvp.second[i], run_combo_type);
        }
        allv[svvp.first] = v;
      }
      return allv;
    }

    /**
     * Prints results at varying levels of verbosity.
     * The behavior of this method depends on whether lambdas_ is set.
     * If lambdas IS set (i.e., this was used to choose lambdas), print:
     *  - 0: best lambda vec, its mean/stderr for main score
     *  - 1: best lambda's mean/stderr for all scores
     *  - 2: all lambdas, means/stderrs for main score
     *  - 3: all lambdas, means/stderrs for all scores
     *  - higher: everything
     * If lambdas IS NOT set (i.e., this was used for testing only), print:
     *  - 0: mean/stderr for main score
     *  - 1: mean/stderr for all scores
     *  - higher: everything
     *
     * @param level  How much info to print. (See above.)
     */
    void print(std::ostream& out, size_t level) const {
      typedef std::pair<std::string, vec> string_vec_pair;
      if (lambdas_.size() != 0) {
        assert(single_results_.size() == lambdas_.size());
        size_t best_i = max_index(single_results_);
        out << "Best lambda: " << lambdas_[best_i] << "\n"
            << "  " << statistics::generalized_mean_string(run_combo_type)
            << ": " << single_results_[best_i] << "\n"
            << "  " << statistics::generalized_deviation_string(run_combo_type)
            << ": " << generalized_deviation(results_[best_i],run_combo_type)
            <<"\n";
        if (level >= 1) {
          foreach(const string_vec_pair& svp, single_all_results_) {
            assert(best_i < svp.second.size());
            assert(all_results_.count(svp.first) &&
                   best_i < safe_get(all_results_,svp.first).size());
            out << "  " << svp.first << " "
                << statistics::generalized_mean_string(run_combo_type)
                << ": " << svp.second[best_i] << "\n"
                << "  " << svp.first << " "
                << statistics::generalized_deviation_string(run_combo_type)
                << ": "
                << generalized_deviation((safe_get(all_results_,svp.first))[best_i],
                                         run_combo_type)
                << "\n";
          }
        }
        if (level >= 2) {
          out << "Summary for all lambdas:\n"
              << "lambda\t"
              << statistics::generalized_mean_string(run_combo_type) << "\t"
              << statistics::generalized_deviation_string(run_combo_type)
              << "\n";
          for (size_t i = 0; i < lambdas_.size(); ++i) {
            out << lambdas_[i] << "\t" << single_results_[i] << "\t"
                << generalized_deviation(results_[i], run_combo_type) << "\n";
          }
        }
        if (level >= 3) {
          assert(false); // NOT YET IMPLEMENTED
        }
      } else {
        assert(single_results_.size() == 1 &&
               results_.size() == 1);
        out << statistics::generalized_mean_string(run_combo_type)
            << ": " << single_results_[0] << "\n"
            << statistics::generalized_deviation_string(run_combo_type)
            << ": " << generalized_deviation(results_[0], run_combo_type)
            << "\n";
        if (level >= 1) {
          foreach(const string_vec_pair& svp, single_all_results_) {
            assert(svp.second.size() == 1);
            assert(all_results_.count(svp.first) &&
                   safe_get(all_results_,svp.first).size() == 1);
            out << svp.first << " "
                << statistics::generalized_mean_string(run_combo_type)
                << ": " << svp.second[0] << "\n"
                << svp.first << " "
                << statistics::generalized_deviation_string(run_combo_type)
                << ": "
                << generalized_deviation(safe_get(all_results_,svp.first)[0],
                                         run_combo_type)
                << "\n";
          }
        }
        if (level >= 2) {
          assert(false); // NOT YET IMPLEMENTED
        }
      }
    } // print

    // Protected data
    // =========================================================================
  protected:

    build_type_enum build_type_;

    //! Copy from cv_params from constructor.
    statistics::generalized_mean_enum run_combo_type;

    boost::mt11213b rng;

    boost::uniform_int<int> unif_int;

    //! (For choosing lambda)
    //! List of all model parameters tested.
    std::vector<vec> lambdas_;

    /**
     * (For choosing lambda)
     *   results_[i] = list of results for each test run for lambdas_[i]
     *
     * (For testing, not choosing lambda)
     *   results_[0] = list of results for each test run
     */
    std::vector<vec> results_;

    /**
     * (For choosing lambda)
     *   all_results_[result type][i]
     *      = list of results for each test run for lambdas_[i]
     *
     * (For testing, not choosing lambda)
     *   all_results_[result type][0] = list of results for each test run
     */
    std::map<std::string, std::vector<vec> > all_results_;

    //! single_results_[i] = mean (or median, etc.) of results_[i]
    vec single_results_;

    //! single_all_results_[result type][i] =
    //!        mean (or median, etc.) of all_results_[result_type][i]
    std::map<std::string, vec> single_all_results_;

    // Protected methods
    // =========================================================================

    //! (For choosing lambda)
    //! Update lambdas_, single_results_, single_all_results_.
    void
    add_single_results(size_t zoom_i, const std::vector<vec>& lambdas_zoom,
                       const param_list_validation_functor<la_type>& val_func) {
      if (zoom_i == 0) {
        lambdas_ = lambdas_zoom;
        single_results_ = val_func.results;
        single_all_results_ = val_func.all_results;
      } else {
        size_t oldsize(lambdas_.size());
        lambdas_.resize(oldsize + lambdas_zoom.size());
        single_results_.reshape(oldsize + lambdas_zoom.size(), 1);
        assert(lambdas_zoom.size() == val_func.results.size());
        for (size_t j(0); j < lambdas_zoom.size(); ++j) {
          lambdas_[oldsize + j] = lambdas_zoom[j];
          single_results_[oldsize + j] = val_func.results[j];
        }
        typedef std::pair<std::string, vec> string_vec_pair;
        foreach(const string_vec_pair& svp, val_func.all_results) {
          assert(lambdas_zoom.size() == svp.second.size());
          single_all_results_[svp.first].reshape
            (oldsize + lambdas_zoom.size(), 1);
          for (size_t j(0); j < lambdas_zoom.size(); ++j) {
            single_all_results_[svp.first][oldsize + j] = svp.second[j];
          }
        }
      }
    } // add_single_results

    //! (For choosing lambda)
    //! Update lambdas_, results_, all_results_.
    void
    add_results(size_t zoom_i, const std::vector<vec>& lambdas_zoom,
                const param_list_validation_functor<la_type>& val_func,
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
          results_[oldsize + i].zeros(num_runs);
          results_[oldsize + i][run_i] = val_func.results[i];
        }
        foreach(const string_vec_pair& svp, val_func.all_results) {
          assert(lambdas_zoom.size() == svp.second.size());
          all_results_[svp.first].resize(oldsize + lambdas_zoom.size());
          for (size_t i = 0; i < lambdas_zoom.size(); ++i) {
            all_results_[svp.first][oldsize + i].zeros(num_runs);
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

    //! (For testing, not choosing lambda)
    //! Update single_results_, single_all_results_.
    //! @param res   Main result value.
    void add_results(const model_validation_functor<la_type>& mv_func,
                     value_type res, size_t fold_i, size_t nfolds) {
      typedef std::pair<std::string, value_type> result_value_pair;
      assert(fold_i < nfolds);
      if (fold_i == 0) {
        results_.resize(1);
        results_[0].zeros(nfolds);
      } else {
        assert(results_.size() == 1 && results_[0].size() == nfolds);
      }
      results_[0][fold_i] = res;
      foreach(const result_value_pair& rvp, mv_func.result_map()) {
        if (fold_i == 0) {
          all_results_[rvp.first].resize(1);
          all_results_[rvp.first][0].zeros(nfolds);
        } else {
          assert(all_results_[rvp.first].size() == 1);
          assert(all_results_[rvp.first][0].size() == nfolds);
        }
        all_results_[rvp.first][0][fold_i] = rvp.second;
      }
    }

    //! (For choosing lambda)
    //!  Combine results_ into single_results_
    //!  and all_results_ into single_all_results_.
    void combine_results() {
      single_results_.set_size(results_.size());
      for (size_t i = 0; i < results_.size(); ++i) {
        single_results_[i] =
          generalized_mean(results_[i], run_combo_type);
      }
      single_all_results_.clear();
      foreach(const std::string& s, keys(all_results_)) {
        single_all_results_[s].set_size(results_.size());
        for (size_t i = 0; i < results_.size(); ++i) {
          single_all_results_[s][i] = 
            generalized_mean(all_results_[s][i], run_combo_type);
        }
      }
    } // combine_results

  }; // class validation_framework

  // =========================================================================
  // Implementations of templated methods in validation_framework
  // =========================================================================

  // Choose parameters via CV.
  template <typename LA>
  validation_framework<LA>::
  validation_framework(const dataset<la_type>& ds,
                       const crossval_parameters& cv_params_,
                       model_validation_functor<la_type>& mv_func,
                       unsigned random_seed)
    : build_type_(CHOOSE_PARAM_CV), run_combo_type(cv_params_.run_combo_type),
      rng(random_seed), unif_int(0, std::numeric_limits<int>::max()) {

    crossval_parameters cv_params(cv_params_);

    if (!cv_params.valid()) {
      throw std::invalid_argument
        ("validation_framework constructed with invalid cv_params.");
    }
    if (cv_params.nfolds > ds.size())
      cv_params.nfolds = ds.size();

    // These hold values for each round of zooming:
    std::vector<vec> lambdas_zoom =
      create_parameter_grid(cv_params.minvals, cv_params.maxvals,
                            cv_params.nvals, cv_params.log_scale);

    param_list_validation_functor<la_type> val_func;

    // Set up dataset views.
    dataset_view<la_type> permuted_view(ds);
    permuted_view.set_record_indices(randperm(ds.size(), rng));
    dataset_view<la_type> fold_train(permuted_view);
    dataset_view<la_type> fold_test(permuted_view);
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
                      mv_func);
        // Add to lambdas_, results_, all_results_.
        add_results(zoom_i, lambdas_zoom, val_func, fold, cv_params.nfolds);
      }

      // Combine results_ into single_results_
      // and all_results_ into single_all_results_.
      combine_results();

      // Prepare next lambdas_zoom.
      if (zoom_i + 1 <= cv_params.zoom) {
        size_t best_i = max_index(single_results_);
        lambdas_zoom =
          zoom_parameter_grid(lambdas_, lambdas_[best_i], cv_params.nvals,
                              cv_params.log_scale);
      }
    } // loop over zooms
  } // constructor (choose parameters via CV)

  // Choose parameters via validation set.
  template <typename LA>
  validation_framework<LA>::
  validation_framework(const dataset<la_type>& train_ds,
                       const dataset<la_type>& test_ds,
                       const crossval_parameters& cv_params,
                       model_validation_functor<la_type>& mv_func,
                       unsigned random_seed)
    : build_type_(CHOOSE_PARAM_VALIDATION),
      run_combo_type(cv_params.run_combo_type),
      rng(random_seed), unif_int(0, std::numeric_limits<int>::max()) {

    if (!cv_params.valid()) {
      throw std::invalid_argument
        ("validation_framework constructed with invalid cv_params.");
    }

    // These hold values for each round of zooming:
    std::vector<vec> lambdas_zoom =
      create_parameter_grid(cv_params.minvals, cv_params.maxvals,
                            cv_params.nvals, cv_params.log_scale);

    param_list_validation_functor<la_type> val_func;

    for (size_t zoom_i(0); zoom_i <= cv_params.zoom; ++zoom_i) {
      // Test lambdas_zoom.
      val_func.test(lambdas_zoom, train_ds, test_ds, unif_int(rng), mv_func);
      // Add to lambdas_, single_results_, single_all_results_.
      add_single_results(zoom_i, lambdas_zoom, val_func);
      // Prepare next lambdas_zoom.
      if (zoom_i + 1 <= cv_params.zoom) {
        size_t best_i = max_index(single_results_);
        lambdas_zoom =
          zoom_parameter_grid(lambdas_, lambdas_[best_i], cv_params.nvals,
                              cv_params.log_scale);
      }
    } // loop over zooms
  } // constructor (choose parameters via validation set)

  template <typename LA>
  validation_framework<LA>::
  validation_framework(const dataset<la_type>& ds,
                       model_validation_functor<la_type>& mv_func,
                       size_t k,
                       unsigned random_seed)
    : build_type_(TEST_CV), run_combo_type(statistics::MEAN),
      rng(random_seed), unif_int(0, std::numeric_limits<int>::max()) {

    assert(k > 1);

    // Set up dataset views.
    dataset_view<la_type> permuted_view(ds);
    permuted_view.set_record_indices(randperm(ds.size(), rng));
    dataset_view<la_type> fold_train(permuted_view);
    dataset_view<la_type> fold_test(permuted_view);
    fold_train.save_record_view();
    fold_test.save_record_view();

    // For each fold of CV,
    for (size_t fold(0); fold < k; ++fold) {
      if (fold != 0) {
        fold_train.restore_record_view();
        fold_test.restore_record_view();
      }
      fold_train.set_cross_validation_fold(fold, k, false);
      fold_test.set_cross_validation_fold(fold, k, true);
      // Test lambdas_zoom.
      value_type res = mv_func.train(fold_train, fold_test, unif_int(rng));
      // Add to results_, all_results_.
      add_results(mv_func, res, fold, k);
    }

    // Combine results_ into single_results_
    // and all_results_ into single_all_results_.
    combine_results();

  } // constructor (test via CV)

  template <typename LA>
  validation_framework<LA>::
  validation_framework(const dataset<la_type>& train_ds,
                       const dataset<la_type>& test_ds,
                       model_validation_functor<la_type>& mv_func,
                       unsigned random_seed)
    : build_type_(TEST_VALIDATION), run_combo_type(statistics::MEAN),
      rng(random_seed), unif_int(0, std::numeric_limits<int>::max()) {

    /*
    - For iterations
       - Run learner
     */

    throw std::runtime_error("validation_framework constructor (test via validation set) not yet implemented.");

  } // constructor (test via validation set)

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_VALIDATION_FRAMEWORK_HPP
