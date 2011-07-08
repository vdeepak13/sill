#include <limits>
#include <stdexcept>

#include <sill/learning/discriminative/linear_regression.hpp>
#include <sill/learning/validation/parameter_grid.hpp>
#include <sill/math/permutations.hpp>
#include <sill/math/statistics.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  std::ostream&
  operator<<(std::ostream& out, const linear_regression_parameters& params) {
    out << "init_iterations: " << params.init_iterations << "\n"
        << "objective: " << params.objective << "\n"
        << "regularization: " << params.regularization << "\n"
        << "lambda: " << params.lambda << "\n"
        << "opt_method: " << params.opt_method << "\n"
        << "perturb_init: " << params.perturb_init << "\n"
        << "convergence_zero: " << params.convergence_zero << "\n"
        << "debug: " << params.debug << "\n";
    return out;
  }

  // Protected methods
  //==========================================================================

  const mat& linear_regression::Ydata() const {
    if (Ydata_ptr)
      return *Ydata_ptr;
    else
      return Ydata_own;
  }

  const mat& linear_regression::Xdata() const {
    if (Xdata_ptr)
      return *Xdata_ptr;
    else
      return Xdata_own;
  }

  void linear_regression::train_matrix_inversion() {
    if (Ydata_ptr) {
      Ydata_own = *Ydata_ptr;
    }
    weights_.b = sum(Ydata_own, 1);
    weights_.b /= Ydata_own.n_rows;
    Ydata_own -= repmat(trans(weights_.b), Ydata().n_rows, 1);
    bool result(false);
    switch(params.regularization) {
    case 0: // none
      result =
        solve(weights_.A,
              trans(Xdata()) * Xdata(), trans(Xdata()) * Ydata_own);
//        ls_solve_chol(trans(Xdata()) * Xdata(),
//                      trans(Xdata()) * Ydata_own, weights_.A);
      break;
    case 2: // L2
      result =
        solve(weights_.A,
              trans(Xdata()) * Xdata()
              + diagmat(.5 * params.lambda * ones<vec>(Xdata().n_cols)),
              trans(Xdata()) * Ydata_own);
//        ls_solve_chol(trans(Xdata()) * Xdata()
//                      + (.5*params.lambda) * eye(Xdata().n_cols,Xdata().n_cols),
//                      trans(Xdata()) * Ydata_own, weights_.A);
      break;
    default:
      assert(false);
    }
    weights_.A = trans(weights_.A);
    if (!result) {
      throw std::runtime_error
        ("Cholesky decomposition failed in linear_regression::train_matrix_inversion");
    }
    if (Ydata_ptr) {
      Ydata_own.set_size(1,1); //set to size 1 instead of 0 since IT++ has issues
      // About these IT++ issues: resize(0,0) will free the internal data in
      // the IT++ matrix, but when the IT++ matrix object is freed, its
      // destructor will try to free the internal data again.
    }
  }

  void linear_regression::train_matrix_inversion_with_mean() {
    /*
    if (!Ydata_ptr) {
      Xdata_own.reshape(Xdata_own.n_rows, Xdata_own.n_cols + 1);
      Xdata_own.set_column(Xdata_own.n_cols - 1, vec(Xdata_own.n_rows, 1.));
    }
    */
    mat tmpA;
    bool result(false);
    switch(params.regularization) {
    case 0: // none
      result = solve(tmpA, trans(Xdata()) * Xdata(), trans(Xdata()) * Ydata());
//        ls_solve_chol(trans(Xdata()) * Xdata(),
//                      trans(Xdata()) * Ydata(), tmpA);
      break;
    case 2: // L2
      result =
        solve(tmpA,
              trans(Xdata()) * Xdata()
              + diagmat(.5 * params.lambda * ones<vec>(Xdata().n_cols)),
              trans(Xdata()) * Ydata());
//        ls_solve_chol(trans(Xdata()) * Xdata()
//                      + (.5 * params.lambda) * eye(Xdata().n_cols,Xdata().n_cols),
//                      trans(Xdata()) * Ydata(), tmpA);
      break;
    default:
      assert(false);
    }
    if (!result) {
      throw std::runtime_error
        ("Cholesky decomposition failed in linear_regression::train_matrix_inversion_with_mean");
    }
    weights_.A = trans(tmpA.rows(0, tmpA.n_rows - 2));
    weights_.b = tmpA.row(tmpA.n_rows - 1);
  } // train_matrix_inversion_with_mean()

  void linear_regression::init(const dataset<la_type>& ds, bool own_data) {
    if (!params.valid()) {
      std::cerr << "linear_regression given invalid parameters:\n"
                << params << std::endl;
      throw std::invalid_argument("bad parameters");
    }
    assert(Yvec.size() > 0);

    rng.seed(static_cast<unsigned>(params.random_seed));

    if (own_data) {
      // Load data into Ydata, Xdata
      ds.get_value_matrix(Ydata_own, Yvec);
      if ((params.opt_method == 0) && params.regularize_mean)
        ds.get_value_matrix(Xdata_own, Xvec, true); // add ones vector
      else
        ds.get_value_matrix(Xdata_own, Xvec, false);
    } else {
      assert(Ydata().n_rows == ds.size());
      assert(Ydata().n_cols == Yvec_size);
      assert(Xdata().n_rows == ds.size());
      if ((params.opt_method == 0) && params.regularize_mean) {
        if (Xdata().n_cols == Xvec_size) {
          // add ones vector
          Xdata_own.set_size(Xdata().n_rows, Xdata().n_cols + 1);
          Xdata_own(span(0,Xdata().n_rows-1), span(0,Xdata().n_cols-1)) =
            Xdata();
          Xdata_own.col(Xdata().n_cols).fill(1);
//          Xdata_own.set_submatrix(0, 0, Xdata());
          Xdata_ptr = NULL;
        } else {
          assert(Xdata().n_cols == Xvec_size + 1);
        }
      } else {
        assert(Xdata().n_cols == Xvec_size);
      }
    }

    // Initialize weights
    if (params.opt_method != 0) {
      weights_.A.set_size(Yvec_size, Xvec_size);
      weights_.b.set_size(Yvec_size);
      if (params.perturb_init > 0) {
        boost::uniform_real<double> uniform_dist;
        uniform_dist = boost::uniform_real<double>
          (-1 * params.perturb_init, params.perturb_init);
        for (size_t i = 0; i < weights_.A.n_rows; ++i)
          for (size_t j = 0; j < weights_.A.n_cols; ++j)
            weights_.A(i,j) = uniform_dist(rng);
        foreach(double& v, weights_.b)
          v = uniform_dist(rng);
      } else {
        weights_ = 0.;
      }
    }

    if (params.opt_method != 0) {
      if (ds.is_weighted()) {
        data_weights = ds.weights();
        total_train_weight = sum(data_weights);
      } else {
        data_weights.set_size(0);
        total_train_weight = ds.size();
      }
    }

    // Initialize for whatever learning algorithm
    switch(params.opt_method) {
    case 0: // Matrix inversion
      if (ds.is_weighted()) {
        std::cerr << "Matrix inversion does not currently support weighted"
                  << " datasets." << std::endl;
        assert(false);
      }
      if (params.regularize_mean)
        train_matrix_inversion_with_mean();
      else
        train_matrix_inversion();
      if (own_data) {
        Ydata_own.set_size(0,0);
        Xdata_own.set_size(0,0);
      }
      break;
    case 1: // Batch gradient descent
      {
        obj_functor_ptr = new objective_functor(*this);
        grad_functor_ptr = new gradient_functor(*this);
        gradient_descent_parameters ga_params;
        ga_params.convergence_zero = params.convergence_zero;
//        if (params.debug > 1)
//          ga_params.debug = params.debug - 1;
        gradient_descent_ptr =
          new gradient_descent<opt_vector,objective_functor,gradient_functor>
          (*obj_functor_ptr, *grad_functor_ptr, weights_, ga_params);
      }
      break;
    case 2: // Batch conjugate gradient
      {
        obj_functor_ptr = new objective_functor(*this);
        grad_functor_ptr = new gradient_functor(*this);
        conjugate_gradient_parameters cg_params;
        cg_params.convergence_zero = params.convergence_zero;
        if (params.debug > 1)
          cg_params.debug = params.debug - 1;
        conjugate_gradient_ptr = new
          conjugate_gradient<opt_vector,objective_functor,gradient_functor>
          (*obj_functor_ptr, *grad_functor_ptr, weights_, cg_params);
      }
      break;
    default:
      assert(false);
    }

    tmpx.set_size(weights_.A.n_cols);

    if (params.opt_method != 0) {
      while (iteration_ < params.init_iterations)
        if (!step())
          break;
    }
  } // end of function init()

  bool linear_regression::step_gradient_descent() {
    if (!gradient_descent_ptr)
      return false;

    double prev_train_obj(train_obj);
    if (!gradient_descent_ptr->step())
      return false;
    train_obj = gradient_descent_ptr->objective();
    if (params.debug > 0) {
      if (train_obj > prev_train_obj)
        std::cerr << "linear_regression took a step which "
                  << "increased the objective from " << prev_train_obj
                  << " to " << train_obj
                  << std::endl;
      std::cerr << "change in objective = "
                << (train_obj - prev_train_obj) << std::endl;
    }
    // Check for convergence
    if (fabs(train_obj - prev_train_obj) < params.convergence_zero) {
      if (params.debug > 0) {
        std::cerr << "linear_regression converged:"
                  << " training objective changed from "
                  << prev_train_obj << " to " << train_obj
                  << "; exiting early (iteration " << iteration() << ")."
                  << std::endl;
        return false;
      }
    }

    ++iteration_;
    return true;
  } // end of function: step_gradient_descent()

  bool linear_regression::step_conjugate_gradient() {
    if (!conjugate_gradient_ptr)
      return false;

    double prev_train_obj(train_obj);
    if (!conjugate_gradient_ptr->step())
      return false;
    train_obj = conjugate_gradient_ptr->objective();
    if (params.debug > 0) {
      if (train_obj > prev_train_obj)
        std::cerr << "linear_regression took a step which "
                  << "increased the objective from " << prev_train_obj
                  << " to " << train_obj
                  << std::endl;
      std::cerr << "change in objective = "
                << (train_obj - prev_train_obj) << std::endl;
    }
    // Check for convergence
    if (fabs(train_obj - prev_train_obj) < params.convergence_zero) {
      if (params.debug > 0) {
        std::cerr << "linear_regression converged:"
                  << " training objective changed from "
                  << prev_train_obj << " to " << train_obj
                  << "; exiting early (iteration " << iteration() << ")."
                  << std::endl;
        return false;
      }
    }

    ++iteration_;
    return true;
  } // end of function: step_conjugate_gradient()

  // Methods for iterative learners
  //==========================================================================

    bool linear_regression::step() {
      switch(params.opt_method) {
      case 0:
        throw std::runtime_error
          ("linear_regression::step() called for non-iterative learning method.");
      case 1:
        return step_gradient_descent();
      case 2:
        return step_conjugate_gradient();
      default:
        assert(false);
      }
      return false;
    }

  // Getters and helpers
  //==========================================================================

  template <>
  vector_domain
  linear_regression::get_dependencies<vector_variable>() const {
    vector_domain x;
    for (size_t i(0); i < Xvec.size(); ++i) {
      if (norm(weights_.A.col(i),1) > params.convergence_zero)
        x.insert(Xvec[i]);
    }
    return x;
  }

  template <>
  std::vector<std::pair<vector_variable*, double> >
  linear_regression::get_dependencies<vector_variable>(size_t K) const {
    if (K == 0)
      K = std::numeric_limits<size_t>::max();
    mutable_queue<vector_variable*, double> vqueue;
    for (size_t i(0); i < Xvec.size(); ++i) {
      vqueue.push(Xvec[i], norm(weights_.A.col(i),1));
    }
    std::vector<std::pair<vector_variable*, double> > x;
    while ((x.size() < K) && (vqueue.size() > 0)) {
      x.push_back(vqueue.top());
      vqueue.pop();
    }
    return x;
  }

  // Methods for choosing regularization
  //==========================================================================

  double
  linear_regression::choose_lambda_easy
  (const vector_var_vector& Yvec, const vector_var_vector& Xvec,
   const linear_regression_parameters& lr_params,
   const dataset<la_type>& ds, unsigned random_seed) {
    size_t n_folds = 10;
    size_t n_lambdas = 10;
    double MIN_LAMBDA = .001;
    double MAX_LAMBDA = ds.size();
    vec lambdas(create_parameter_grid(MIN_LAMBDA, MAX_LAMBDA, n_lambdas,
                                      true, true));
    vec all_lambdas, scores, stderrs;
    // See if we can do closed form computations for LOOCV errors.
    if ((lr_params.objective == 2) && (lr_params.regularization == 2)) {
      std::pair<double, linear_regression*> lambda_choice
        = choose_lambda_ridge(all_lambdas, scores, stderrs, Yvec, Xvec, lambdas,
                              lr_params, false, ds, false, random_seed);
      return lambda_choice.first;
    }
    // Otherwise, do CV the hard way.
    return choose_lambda_cv(all_lambdas, scores, stderrs, Yvec, Xvec, n_folds,
                            lambdas, lr_params, false, ds, random_seed);
  }

  double
  linear_regression::choose_lambda_easy
  (const linear_regression_parameters& lr_params,
   const dataset<la_type>& ds, unsigned random_seed) {
    vector_var_vector Yvec(ds.vector_class_variables());
    vector_domain Yset(Yvec.begin(), Yvec.end());
    vector_var_vector Xvec;
    foreach(vector_variable* v, ds.vector_variables()) {
      if (Yset.count(v) == 0)
        Xvec.push_back(v);
    }
    return choose_lambda_easy(Yvec, Xvec, lr_params, ds, random_seed);
  }

  double
  linear_regression::choose_lambda_cv
  (vec& all_lambdas, vec& scores, vec& stderrs,
   const vector_var_vector& Yvec, const vector_var_vector& Xvec,
   size_t n_folds, const vec& lambdas,
   const linear_regression_parameters& lr_params,
   size_t zoom, const dataset<la_type>& ds, unsigned random_seed) {

    if (lr_params.regularization == 0) {
      std::cerr << "linear_regression::choose_lambda_cv() was called when "
                << "lr_params.regularization = 0." << std::endl;
      return 0.;
    }

    assert((n_folds > 0) && (n_folds <= ds.size()));
    assert(lambdas.size() > 0);

    all_lambdas.set_size(0);
    scores.set_size(0);
    stderrs.set_size(0);
    vec lambdas_zoom(lambdas); // These hold values for each round of zooming.
    vec scores_zoom(lambdas.size(), 0);
    vec stderrs_zoom(lambdas.size(), 0);
    boost::mt11213b rng(random_seed);
    size_t best_i(0); // This indexes the current best value in all_lambdas.

    dataset_view<la_type> permuted_view(ds);
    permuted_view.set_record_indices(randperm(ds.size(), rng));
    dataset_view<la_type> fold_train_view(permuted_view);
    dataset_view<la_type> fold_test_view(permuted_view);
    fold_train_view.save_record_view();
    fold_test_view.save_record_view();
    linear_regression_parameters fold_params(lr_params);
    mat Ydata, Xdata;
    mat temp_scores_mad;
    for (size_t zoom_i(0); zoom_i <= zoom; ++zoom_i) {
      if (lr_params.cv_score_type == 1)
        temp_scores_mad.set_size(n_folds, lambdas_zoom.size());
      for (size_t fold(0); fold < n_folds; ++fold) {
        // Prepare the fold dataset views
        if (fold != 0) {
          fold_train_view.restore_record_view();
          fold_test_view.restore_record_view();
        }
        fold_train_view.set_cross_validation_fold(fold, n_folds, false);
        fold_test_view.set_cross_validation_fold(fold, n_folds, true);
        fold_train_view.get_value_matrix(Ydata, Yvec);
        if ((lr_params.opt_method == 0) && lr_params.regularize_mean)
          fold_train_view.get_value_matrix(Xdata, Xvec, true); // add ones vec
        else
          fold_train_view.get_value_matrix(Xdata, Xvec, false);
        for (size_t k(0); k < lambdas_zoom.size(); ++k) {
          fold_params.lambda = lambdas_zoom[k];
          linear_regression
            lr(fold_train_view, Yvec, Xvec, Ydata, Xdata, fold_params);
          double tmpval(lr.mean_squared_error(fold_test_view).first);
          if (lr_params.cv_score_type != 1) {
            scores_zoom[k] += tmpval;
            stderrs_zoom[k] += tmpval * tmpval;
          } else {
            temp_scores_mad(fold, k) = tmpval;
          }
        }
      }
      if (lr_params.cv_score_type == 1) {
        for (size_t k(0); k < lambdas_zoom.size(); ++k) {
          std::pair<double,double> m_mad(median_MAD(vec(temp_scores_mad.col(k)))); // TO DO: AVOID COPY
          scores_zoom[k] = m_mad.first;
          stderrs_zoom[k] = m_mad.second;
        }
      }
      size_t tmp_best_i(min_index(scores_zoom, rng));
      if (zoom_i == 0) {
        best_i = tmp_best_i;
        all_lambdas = lambdas_zoom;
        scores = scores_zoom;
        stderrs = stderrs_zoom;
      } else {
        size_t old_size(all_lambdas.size());
        if (scores_zoom[tmp_best_i] < scores[best_i])
          best_i = old_size + tmp_best_i;
        all_lambdas.reshape(old_size + lambdas_zoom.size(), 1);
        scores.reshape(old_size + scores_zoom.size(), 1);
        stderrs.reshape(old_size + stderrs_zoom.size(), 1);
        for (size_t k(0); k < lambdas_zoom.size(); ++k) {
          all_lambdas[old_size + k] = lambdas_zoom[k];
          scores[old_size + k] = scores_zoom[k];
          stderrs[old_size + k] = stderrs_zoom[k];
        }
      }
      if (zoom_i != zoom) {
        lambdas_zoom =
          zoom_parameter_grid(all_lambdas, all_lambdas[best_i],
                              lambdas.size(), lr_params.cv_log_scale);
        if (scores_zoom.size() != lambdas_zoom.size()) {
          scores_zoom.set_size(lambdas_zoom.size());
          stderrs_zoom.set_size(lambdas_zoom.size());
        }
        scores_zoom.zeros();
        stderrs_zoom.zeros();
      }
    } // loop over zooming iterations

    if (lr_params.cv_score_type != 1) {
      for (size_t k(0); k < scores.size(); ++k) {
        if (scores[k] == std::numeric_limits<double>::infinity())
          continue;
        scores[k] = scores[k] / n_folds;
        stderrs[k] = std::sqrt((stderrs[k]/n_folds) - (scores[k] * scores[k]));
      }
    }
    return all_lambdas[best_i];
  } // end of function choose_lambda_cv()

  std::pair<double, linear_regression*>
  linear_regression::choose_lambda_ridge
  (vec& all_lambdas, vec& scores, vec& stderrs,
   const vector_var_vector& Yvec, const vector_var_vector& Xvec,
   const vec& lambdas, const linear_regression_parameters& lr_params,
   size_t zoom, const dataset<la_type>& ds, bool return_regressor,
   unsigned random_seed) {

    if (lr_params.regularization == 0) {
      std::cerr << "linear_regression::choose_lambda_cv() was called when "
                << "lr_params.regularization = 0." << std::endl;
      return std::make_pair(0., static_cast<linear_regression*>(NULL));
    }

    assert(ds.size() > 0);
    assert(lambdas.size() > 0);
    assert(lr_params.regularization == 2);

    all_lambdas.set_size(0);
    scores.set_size(0);
    stderrs.set_size(0);
    vec lambdas_zoom(lambdas); // These hold values for each round of zooming.
    vec scores_zoom(lambdas.size(), 0);
    vec stderrs_zoom(lambdas.size(), 0);
    boost::mt11213b rng(random_seed);
    size_t best_i(0); // This indexes the current best value in all_lambdas.

    mat Ydata;
    mat Xdata;
    ds.get_value_matrix(Ydata, Yvec);
    vec mean_b;
    if (lr_params.regularize_mean) {
      ds.get_value_matrix(Xdata, Xvec, true);
    } else {
      mean_b = sum(Ydata, 1);
      mean_b /= Ydata.n_rows;
      Ydata -= repmat(trans(mean_b), Ydata.n_rows, 1);
      ds.get_value_matrix(Xdata, Xvec, false);
    }
    mat G0(trans(Xdata) * Xdata);
    // Compute G0 = U D V, where diag(D) = s0.
    mat Ut;
    mat Vt;
    vec s0;
    bool result = svd(Ut, s0, Vt, G0);
    if (!result) {
      // TO DO: I should do inversion with lambda > 0 in this case, rather
      //        than throwing an error.
      throw std::runtime_error("SVD failed in linear_regression_choose_lambda()...but this is fixable.");
    }
    Ut = trans(Ut);
    bool s0_has_0(min(s0) == 0);
    vec temp_scores_mad; // for computing median/MAD
    if (lr_params.cv_score_type == 1)
      temp_scores_mad.set_size(Xdata.n_rows);
    for (size_t zoom_i(0); zoom_i <= zoom; ++zoom_i) {
      for (size_t k(0); k < lambdas_zoom.size(); ++k) {
        if (s0_has_0 && lambdas_zoom[k] == 0.) {
          scores_zoom[k] = std::numeric_limits<double>::infinity();
          stderrs_zoom[k] = std::numeric_limits<double>::infinity();
          continue;
        }
        mat X_Glambda_inv_Xt(Xdata * Vt
                             * diagmat(1. / (s0 + .5 * lambdas_zoom[k]))
                             * Ut * trans(Xdata));
        mat Ypred(X_Glambda_inv_Xt * Ydata);
        vec bottom_weights(1. - diagvec(X_Glambda_inv_Xt));
        bottom_weights %= bottom_weights;
        Ypred -= Ydata;
        Ypred %= Ypred;
        for (size_t i(0); i < bottom_weights.size(); ++i) {
          double val(sum(Ypred.row(i)));
          if (bottom_weights[i] == 0) {
            scores_zoom[k] = std::numeric_limits<double>::infinity();
            stderrs_zoom[k] = std::numeric_limits<double>::infinity();
            break;
          }
          val /= bottom_weights[i];
          if (lr_params.cv_score_type != 1) {
            scores_zoom[k] += val;
            stderrs_zoom[k] += val * val;
          } else {
            temp_scores_mad[i] = val;
          }
        }
        if (lr_params.cv_score_type == 1) {
          std::pair<double,double> m_mad(median_MAD(temp_scores_mad));
          scores_zoom[k] = m_mad.first;
          stderrs_zoom[k] = m_mad.second;
        }
      }
      size_t tmp_best_i(min_index(scores_zoom, rng));
      if (zoom_i == 0) {
        best_i = tmp_best_i;
        all_lambdas = lambdas_zoom;
        scores = scores_zoom;
        stderrs = stderrs_zoom;
      } else {
        size_t old_size(all_lambdas.size());
        if (scores_zoom[tmp_best_i] < scores[best_i])
          best_i = old_size + tmp_best_i;
        all_lambdas.reshape(old_size + lambdas_zoom.size(), 1);
        scores.reshape(old_size + scores_zoom.size(), 1);
        stderrs.reshape(old_size + stderrs_zoom.size(), 1);
        for (size_t k(0); k < lambdas_zoom.size(); ++k) {
          all_lambdas[old_size + k] = lambdas_zoom[k];
          scores[old_size + k] = scores_zoom[k];
          stderrs[old_size + k] = stderrs_zoom[k];
        }
      }
      if (zoom_i != zoom) {
        lambdas_zoom =
          zoom_parameter_grid(all_lambdas, all_lambdas[best_i],
                              lambdas.size(), lr_params.cv_log_scale);
        if (scores_zoom.size() != lambdas_zoom.size()) {
          scores_zoom.set_size(lambdas_zoom.size());
          stderrs_zoom.set_size(lambdas_zoom.size());
        }
        scores_zoom.zeros();
        stderrs_zoom.zeros();
      }
    } // loop over zooming iterations

    if (lr_params.cv_score_type != 1) {
      scores /= ds.size();
      stderrs /= ds.size();
      stderrs -= scores % scores;
      stderrs /= ds.size();
      stderrs = sqrt(stderrs);
    }

    if (return_regressor) {
      linear_regression* lr_ptr = new linear_regression(Yvec, Xvec);
      double best_lambda = all_lambdas[best_i];
      mat tmpA(Vt * diagmat(1. / (s0 + .5 * best_lambda)) * Ut
               * trans(Xdata) * Ydata);
      if (lr_params.regularize_mean) {
        lr_ptr->weights_.A = trans(tmpA.rows(0, tmpA.n_rows - 2));
        lr_ptr->weights_.b = tmpA.row(tmpA.n_rows - 1);
      } else {
        lr_ptr->weights_.A = trans(tmpA);
        lr_ptr->weights_.b = mean_b;
      }
      return std::make_pair(best_lambda, lr_ptr);
    } else {
      return std::make_pair(all_lambdas[best_i],
                            static_cast<linear_regression*>(NULL));
    }
  } // choose_lambda_ridge()

  // Free functions
  //==========================================================================

  std::ostream&
  operator<<(std::ostream& out, const linear_regression& lr) {
    lr.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
