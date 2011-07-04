
#include <sill/learning/learn_crf_factor.hpp>

#include <sill/macros_def.hpp>


// Macro for specializing learn_crf_factor to hybrid_crf_factor instances.
// (Specialization definition)
#define GEN_LEARN_CRF_FACTOR_HYBRID_DEF(F)                              \
  template <>                                                           \
  hybrid_crf_factor<F>*                                                 \
  learn_crf_factor<hybrid_crf_factor<F> >                               \
  (const dataset<hybrid_crf_factor<F>::la_type>& ds,                    \
   const hybrid_crf_factor<F>::output_domain_type& Y_,                  \
   copy_ptr<hybrid_crf_factor<F>::input_domain_type> X_ptr_,            \
   const hybrid_crf_factor<F>::parameters& params,                      \
   unsigned random_seed) {                                              \
                                                                        \
    typedef variable_type_group<F::input_variable_type>::domain_type    \
      sub_input_domain_type;                                            \
    assert(params.valid());                                             \
    assert(X_ptr_);                                                     \
    assert(includes(*X_ptr_,params.hcf_x2));                                \
                                                                        \
    boost::mt11213b rng(random_seed);                                   \
    boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max()); \
                                                                        \
    std::vector<F*> subfactor_ptrs(num_assignments(params.hcf_x2), NULL);   \
    finite_var_vector x2_vec(params.hcf_x2.begin(), params.hcf_x2.end());       \
    std::vector<size_t>                                                 \
      x2_multipliers(hybrid_crf_factor<F>::compute_multipliers(x2_vec)); \
    copy_ptr<sub_input_domain_type>                                     \
      sub_X_ptr_(new sub_input_domain_type());                          \
    convert_domain<variable,F::input_variable_type>                     \
      (set_difference                                                   \
       (*X_ptr_, domain(params.hcf_x2.begin(), params.hcf_x2.end())),           \
       *sub_X_ptr_);                                                    \
                                                                        \
    foreach(const finite_assignment& x2, assignments(params.hcf_x2)) {  \
      dataset_view<hybrid_crf_factor<F>::la_type> sub_ds(ds);           \
      sub_ds.restrict_to_assignment(x2);                                \
      size_t i =                                                        \
        hybrid_crf_factor<F>::subfactor_index(x2, x2_vec, x2_multipliers); \
      assert(i < subfactor_ptrs.size());                                \
      subfactor_ptrs[i] =                                               \
        learn_crf_factor<F>(sub_ds, Y_, sub_X_ptr_,                     \
                            params, unif_int(rng));                     \
    }                                                                   \
    return new hybrid_crf_factor<F>(Y_, X_ptr_, subfactor_ptrs, x2_vec, \
                                    x2_multipliers);                    \
  }


/**
 * Macro for specializing learn_crf_factor_cv to hybrid_crf_factor instances.
 * (Specialization definition)
 *
 * NOTE: This is different from other specializations in that it chooses
 *       regularization parameters separately for each sub-factor.
 *       Therefore, the returned reg_params, means, and stderrs values
 *       are different; they are concatenated values from all sub-factors.
 */
#define GEN_LEARN_CRF_FACTOR_CV_HYBRID_DEF(F)                           \
  template <>                                                           \
  hybrid_crf_factor<F>*                                                 \
  learn_crf_factor_cv<hybrid_crf_factor<F> >                            \
  (std::vector<hybrid_crf_factor<F>::regularization_type>& reg_params,  \
   vec& means, vec& stderrs,                                            \
   const crossval_parameters& cv_params,                                \
   const dataset<hybrid_crf_factor<F>::la_type>& ds,                    \
   const hybrid_crf_factor<F>::output_domain_type& Y_,                  \
   copy_ptr<hybrid_crf_factor<F>::input_domain_type> X_ptr_,            \
   const hybrid_crf_factor<F>::parameters& params,                      \
   unsigned random_seed) {                                              \
                                                                        \
    typedef variable_type_group<F::input_variable_type>::domain_type    \
      sub_input_domain_type;                                            \
    assert(params.valid());                                             \
    assert(X_ptr_);                                                     \
    assert(includes(*X_ptr_,params.hcf_x2));                                \
                                                                        \
    boost::mt11213b rng(random_seed);                                   \
    boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max()); \
                                                                        \
    reg_params.resize(0);                                               \
    means.resize(0);                                                    \
    stderrs.resize(0);                                                  \
    std::vector<F::regularization_type> sub_reg_params;                 \
    vec sub_means, sub_stderrs;                                         \
                                                                        \
    std::vector<F*> subfactor_ptrs(num_assignments(params.hcf_x2), NULL);   \
    finite_var_vector x2_vec(params.hcf_x2.begin(), params.hcf_x2.end());       \
    std::vector<size_t>                                                 \
      x2_multipliers(hybrid_crf_factor<F>::compute_multipliers(x2_vec)); \
    copy_ptr<sub_input_domain_type>                                     \
      sub_X_ptr_(new sub_input_domain_type());                          \
    convert_domain<variable,F::input_variable_type>                     \
      (set_difference                                                   \
       (*X_ptr_, domain(params.hcf_x2.begin(), params.hcf_x2.end())),           \
       *sub_X_ptr_);                                                    \
                                                                        \
    size_t return_j(0);                                                 \
    foreach(const finite_assignment& x2, assignments(params.hcf_x2)) {  \
      dataset_view<hybrid_crf_factor<F>::la_type> sub_ds(ds);           \
      sub_ds.restrict_to_assignment(x2);                                \
      size_t i =                                                        \
        hybrid_crf_factor<F>::subfactor_index(x2, x2_vec, x2_multipliers); \
      assert(i < subfactor_ptrs.size());                                \
      subfactor_ptrs[i] =                                               \
        learn_crf_factor_cv<F>(sub_reg_params, sub_means, sub_stderrs,  \
                               cv_params, sub_ds, Y_, sub_X_ptr_,       \
                               params, unif_int(rng));                  \
      if (reg_params.size() == 0) {                                     \
        assert(sub_means.size() == sub_stderrs.size() &&                \
               sub_means.size() == sub_reg_params.size());              \
        reg_params.resize(num_assignments(params.hcf_x2) * sub_reg_params.size()); \
        means.resize(reg_params.size());                                \
        stderrs.resize(reg_params.size());                              \
      }                                                                 \
      for (size_t j(0); j < sub_reg_params.size(); ++j) {               \
        reg_params[return_j] = sub_reg_params[j];                       \
        means[return_j] = sub_means[j];                                 \
        stderrs[return_j] = sub_stderrs[j];                             \
        ++return_j;                                                     \
      }                                                                 \
    }                                                                   \
    return new hybrid_crf_factor<F>(Y_, X_ptr_, subfactor_ptrs, x2_vec, \
                                    x2_multipliers);                    \
  }


namespace sill {

  // Definitions: Learning CRF Factors with Fixed Regularization
  //============================================================================

  //! Specialization: table_crf_factor
  template <>
  table_crf_factor*
  learn_crf_factor<table_crf_factor>
  (const dataset<table_crf_factor::la_type>& ds,
   const finite_domain& Y_, copy_ptr<finite_domain> X_ptr_,
   const table_crf_factor::parameters& params, unsigned random_seed) {
    assert(ds.has_variables(Y_));
    assert(X_ptr_);
    assert(ds.has_variables(*X_ptr_));
    assert(params.valid());
    return new table_crf_factor
      (learn_factor<table_factor>::learn_marginal(set_union(Y_, *X_ptr_), ds,
                                                  params.reg.lambdas[0]),
       Y_, false);
  } // learn_crf_factor<table_crf_factor>


  //! Specialization: log_reg_crf_factor
  template <>
  log_reg_crf_factor*
  learn_crf_factor<log_reg_crf_factor>
  (const dataset<log_reg_crf_factor::la_type>& ds,
   const finite_domain& Y_, copy_ptr<domain> X_ptr_,
   const log_reg_crf_factor::parameters& params, unsigned random_seed) {
    if (!ds.has_variables(Y_)) {
      std::cerr << "learn_crf_factor given Y_ = " << Y_
                << ", but the dataset only contains finite variables: "
                << finite_domain(ds.finite_variables().first,
                                 ds.finite_variables().second)
                << std::endl;
      assert(false);
    }
    assert(X_ptr_);

    assert(includes(ds.variables(), *X_ptr_));
    assert(params.valid());
    // Set up dataset view
    dataset_view<log_reg_crf_factor::la_type> ds_view(ds);
    std::set<size_t> finite_indices;
    size_t new_class_var_size(1);
    foreach(finite_variable* v, Y_) {
      finite_indices.insert(ds_view.record_index(v));
      new_class_var_size *= v->size();
    }
    std::set<size_t> vector_indices;
    foreach(variable* v, *X_ptr_) {
      switch(v->get_variable_type()) {
      case variable::FINITE_VARIABLE:
        finite_indices.insert
          (ds_view.record_index(dynamic_cast<finite_variable*>(v)));
        break;
      case variable::VECTOR_VARIABLE:
        vector_indices.insert
          (ds_view.record_index(dynamic_cast<vector_variable*>(v)));
        break;
      default:
        assert(false);
      }
    }
    ds_view.set_variable_indices(finite_indices, vector_indices);
    ds_view.set_finite_class_variables(Y_);
    dataset_statistics<log_reg_crf_factor::la_type> stats(ds_view);
    // Train multilabel logistic regressor
    multiclass_logistic_regression_parameters mlr_params(params.mlr_params);
    mlr_params.regularization = params.reg.regularization;
    mlr_params.lambda = params.reg.lambdas[0];
    finite_variable* new_merged_var
      = params.u.new_finite_variable(new_class_var_size);
    multiclass2multilabel_parameters m2m_params;
    m2m_params.base_learner =
      boost::shared_ptr<multiclass_classifier<> >
      (new multiclass_logistic_regression<>(mlr_params));
    m2m_params.random_seed = random_seed;
    m2m_params.new_label = new_merged_var;
    return new log_reg_crf_factor
      (boost::shared_ptr<multiclass2multilabel>(new multiclass2multilabel
                                                (stats, m2m_params)),
       params.smoothing / ds.size(), Y_, X_ptr_);
  } // learn_crf_factor<log_reg_crf_factor>()


  //! Specialization: gaussian_crf_factor
  template <>
  gaussian_crf_factor*
  learn_crf_factor<gaussian_crf_factor>
  (const dataset<gaussian_crf_factor::la_type>& ds,
   const vector_domain& Y_, copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed) {
    assert(ds.size() > 0);
    assert(includes(ds.variables(), Y_));
    assert(X_ptr_);
    assert(includes(ds.variables(), *X_ptr_));
    assert(params.valid());

    vector_var_vector Yvec(Y_.begin(), Y_.end());
    vector_var_vector Xvec(X_ptr_->begin(), X_ptr_->end());
    mat Ydata; // Y data matrix (nrecords x |Y|)
    ds.get_value_matrix(Ydata, Yvec);
    mat Xdata; // X data matrix (nrecords x |X|)
    ds.get_value_matrix(Xdata, Xvec, false);
    // Compute mean, and center Y values.
    vec mu(sum(Ydata,1));
    mu /= Ydata.n_rows;
    Ydata -= repmat(mu, Ydata.n_rows, 1, true);

    // Compute X'Y
    mat XtY(Xdata.transpose() * Ydata);
    // Compute coefficients
    mat Ct;
    bool result;
    if (params.reg.lambdas[0] > 0)
      result = ls_solve_chol(Xdata.transpose() * Xdata
                             + params.reg.lambdas[0] * identity(Xdata.n_cols),
                             XtY, Ct);
    else
      result = ls_solve_chol(Xdata.transpose() * Xdata, XtY, Ct);
    if (!result) {
      throw std::runtime_error("Cholesky decomposition failed in gaussian_crf_factor::learn_crf_factor().  Try using more regularization.");
    }
    // Compute covariance matrix
    mat cov(Ydata.transpose() * Ydata);
    if (params.reg.lambdas[1] > 0) {
      cov += params.reg.lambdas[1] * identity(cov.n_rows);
    }
    cov -= XtY.transpose() * Ct;
    cov /= ds.size();
    moment_gaussian mg(Yvec, mu, cov, Xvec, Ct.transpose());
    gaussian_crf_factor* gcf_ptr = NULL;
    try {
      gcf_ptr = new gaussian_crf_factor(mg);
    } catch (const inv_error& e) {
      throw;
    } catch (const chol_error& e) {
      if (params.debug)
        std::cerr << "gaussian_crf_factor::learn_crf_factor() failed, probably"
                  << " due to using too little regularization (lambda_bC = "
                  << params.reg.lambdas[0] << ", lambda_cov = "
                  << params.reg.lambdas[1] << ")."
                  << " The covariance matrix was:\n"
                  << cov << std::endl;
      /*
      if (params.max_lambda_cov > params.reg.lambdas[1]) {
        for(size_t lc_inc(1); lc_inc <= params.lambda_cov_increments; ++lc_inc){
          double old_lambda_cov =
            (lc_inc-1.) * ((params.max_lambda_cov - params.reg.lambdas[1])
                           / params.lambda_cov_increments)
            + params.reg.lambdas[1];
          double new_lambda_cov =
            lc_inc * ((params.max_lambda_cov - params.reg.lambdas[1])
                      / params.lambda_cov_increments)
            + params.reg.lambdas[1];
          cov += ((new_lambda_cov - old_lambda_cov) / ds.size())
            * identity(cov.n_rows);
          // FINISH THIS IF NECESSARY
        }
      */
      throw;
    } catch (const ls_solve_chol_error& e) {
      throw;
    }
    return gcf_ptr;
  } // learn_crf_factor<gaussian_crf_factor>()


  GEN_LEARN_CRF_FACTOR_HYBRID_DEF(gaussian_crf_factor)


  // Definitions: Learning CRF Factors with Regularization Chosen via CV
  //============================================================================

  namespace impl {

    gcf_learn_crf_factor_cv_functor::
    gcf_learn_crf_factor_cv_functor
    (const dataset<la_type>& ds,
     const vector_domain& Y_, copy_ptr<vector_domain> X_ptr,
     const gaussian_crf_factor::parameters& params)
      : ds(ds), Y_ptr(&Y_), X_ptr(X_ptr), params_ptr(&params) {
      assert(X_ptr);
    }

    vec gcf_learn_crf_factor_cv_functor::
    operator()(vec& means, vec& stderrs, const std::vector<vec>& lambdas,
               size_t nfolds, unsigned random_seed) const {
      assert((nfolds > 0) && (nfolds <= ds.size()));
      assert(lambdas.size() > 0);
      assert(lambdas[0].size() ==
             gaussian_crf_factor::regularization_type::nlambdas);

    /*
    if ((vector_size(Y_) == 1) && (cv_params.nfolds == ds.size())) {
      // Then we may as well do LOOCV via ridge regression.
      vector_var_vector Yvec(Y_.begin(), Y_.end());
      vector_var_vector Xvec(X_ptr_->begin(), X_ptr_->end());
      vec all_lambdas;
      assert(reg_lambdas.size() == 2);
      vec orig_lambdas(reg_lambdas[0].size());
      for (size_t j(0); j < reg_lambdas[0].size(); ++j)
        orig_lambdas[j] = reg_lambdas[0][j];
      linear_regression_parameters lr_params;
      lr_params.objective = 2;
      lr_params.regularization = 2;
      lr_params.opt_method = 0;
      std::pair<double, linear_regression*> lambda_ridge_result =
        linear_regression::choose_lambda_ridge
        (all_lambdas, means, stderrs, Yvec, Xvec,
         orig_lambdas, lr_params, cv_params.zoom, ds, true, unif_int(rng));
      const linear_regression& lr = *(lambda_ridge_result.second);

      reg_params.resize(all_lambdas.size());
      for (size_t j(0); j < all_lambdas.size(); ++j) {
        reg_params[j].lambdas[0] = all_lambdas[j];
        reg_params[j].lambdas[1] = 0;
      }
      means += vector_size(Yvec) * std::log(2. * pi());
      means *= -.5;
      stderrs *= .5;
      gaussian_crf_factor* tmp_gcf_ptr = new gaussian_crf_factor(lr, ds);
      delete(lambda_ridge_result.second);
      return tmp_gcf_ptr;
    } // end of if (vector_size(Y_) == 1)
    */

      std::vector<vec> alt_lambdas(convert_parameter_grid_to_alt(lambdas));
      assert(alt_lambdas.size() ==
             gaussian_crf_factor::regularization_type::nlambdas);
      // lambdas_index[L] = index in lambdas of value L
      std::map<std::pair<double,double>, size_t> lambdas_index;
      for (size_t i(0); i < lambdas.size(); ++i)
        lambdas_index[std::make_pair(lambdas[i][0],lambdas[i][1])] = i;

      boost::mt11213b rng(random_seed);
      boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
      vector_var_vector Yvec(Y_ptr->begin(), Y_ptr->end());
      vector_var_vector Xvec(X_ptr->begin(), X_ptr->end());

      means.resize(lambdas.size());
      means.zeros_memset();
      stderrs.resize(lambdas.size());
      stderrs.zeros_memset();

      double ll_constant = -.5 * vector_size(Yvec) * std::log(2. * pi());

      dataset_view<la_type> permuted_view(ds);
      permuted_view.set_record_indices(randperm(ds.size(), rng));
      dataset_view<la_type> fold_train_view(permuted_view);
      dataset_view<la_type> fold_test_view(permuted_view);
      fold_train_view.save_record_view();
      fold_test_view.save_record_view();
      mat Xdata; // X data matrix (nrecords x nvars)
      mat Ydata; // Y data matrix
      mat testXdata;
      mat testYdata;
      // For each fold
      for (size_t fold(0); fold < nfolds; ++fold) {
        // Prepare the fold dataset views
        if (fold != 0) {
          fold_train_view.restore_record_view();
          fold_test_view.restore_record_view();
        }
        fold_train_view.set_cross_validation_fold(fold, nfolds, false);
        fold_test_view.set_cross_validation_fold(fold, nfolds, true);
        // Get data
        fold_train_view.get_value_matrix(Xdata, Xvec, false);
        fold_train_view.get_value_matrix(Ydata, Yvec);
        fold_test_view.get_value_matrix(testXdata, Xvec, false);
        fold_test_view.get_value_matrix(testYdata, Yvec);
        // Compute mean, and center Y values.
        vec mu(sum(Ydata,1));
        mu /= Ydata.n_rows;
        Ydata -= repmat(mu, Ydata.n_rows, 1, true);
        testYdata -= repmat(mu, testYdata.n_rows, 1, true);
        // If (# examples) >= |X|, then do SVD of X' X.
        // Otherwise, do SVD of X X'.
        bool num_exs_bigger = (Xdata.n_rows >= Xdata.n_cols);
        // Compute X' * X = U D0 V, where diag(D0) = s0.
        mat tmpmat(num_exs_bigger ?
                   Xdata.transpose() * Xdata :
                   Xdata * Xdata.transpose());
        mat Ut;
        mat Vt;
        vec s0;
        bool result = svd(tmpmat, Ut, s0, Vt);
        if (!result) {
          // TO DO: I should do inversion with lambda > 0 in this case, rather
          //        than throwing an error.
          throw std::runtime_error("SVD failed in gaussian_crf_factor::learn_crf_factor_cv()...but this is fixable.");
        }
        Ut = Ut.transpose();
        mat YtX_n(Ydata.transpose() * Xdata);
        if (num_exs_bigger) {
          Vt = YtX_n * Vt;
        } else {
          Vt = Xdata.transpose() * Vt;
          Ut *= Ydata;
        }
        YtX_n /= fold_train_view.size();
        // Now, coefficients A = Vt * (D0 + \lambda I)^{-1} * Ut.
        bool s0_has_0(min(s0) == 0);

        // Pre-compute Y' * Y / (# training exs)
        mat YtY_n(Ydata.transpose() * Ydata);
        YtY_n /= fold_train_view.size();

        vec tmpvec;
        // For each lambda_bC
        for (size_t i_bC(0); i_bC < alt_lambdas[0].size(); ++i_bC) {
          double lambda_bC(alt_lambdas[0][i_bC]);
          if (s0_has_0 && lambda_bC == 0.) {
            for (size_t i_cov(0); i_cov < alt_lambdas[1].size(); ++i_cov) {
              size_t reg_params_i =
                lambdas_index[std::make_pair
                              (lambda_bC,alt_lambdas[0][i_cov])];
              means[reg_params_i] = std::numeric_limits<double>::infinity();
              stderrs[reg_params_i]= std::numeric_limits<double>::infinity();
            }
            continue;
          }

          mat Amat(Vt * diag(1. / (s0 + lambda_bC)) * Ut);

          // Compute: (Y' Y - Coeff * X'Y) / (# training exs) = yU yD0 yV,
          // where diag(yD0) = ys0.
          mat yUt;
          mat yVt;
          vec ys0;
          tmpmat = YtY_n;
          if (num_exs_bigger) {
            tmpmat -= Amat * YtX_n.transpose();
          } else {
            tmpmat -= YtX_n * Amat;
          }
          bool result = svd(tmpmat, yUt, ys0, yVt);
          if (!result) {
            // TO DO: I should do inversion with lambda > 0 in this case,
            //        rather than throwing an error.
            throw std::runtime_error("SVD failed in gaussian_crf_factor::learn_crf_factor_cv()...but this is fixable too.");
          }
          yUt = yUt.transpose();

          bool ys0_has_0(min(ys0) == 0);
          double logdet_sigma0 = sum(log(ys0));
          if (num_exs_bigger)
            tmpmat = testXdata * Amat.transpose();
          else
            tmpmat = testXdata * Amat;
          tmpmat -= testYdata; // tmpmat = Z A' - Y (for test data)

          // For each lambda_cov
          for (size_t i_cov(0); i_cov < alt_lambdas[1].size(); ++i_cov) {
            double lambda_cov(alt_lambdas[1][i_cov] / fold_train_view.size());
            size_t reg_params_i =
              lambdas_index[std::make_pair(lambda_bC,lambda_cov)];
            if (ys0_has_0 && lambda_cov == 0.) {
              means[reg_params_i] = std::numeric_limits<double>::infinity();
              stderrs[reg_params_i]= std::numeric_limits<double>::infinity();
              continue;
            }
            mat sigma_inv(yVt * diag(1. / (ys0 + lambda_cov)) * yUt);
            double ll(0.);
            for (size_t i(0); i < fold_test_view.size(); ++i) {
              tmpvec = tmpmat.row(i);
              ll += dot(tmpvec, sigma_inv * tmpvec);
            }
            ll *= (-.5 / fold_test_view.size());
            if (lambda_cov == 0)
              ll += ll_constant - .5 * logdet_sigma0;
            else
              ll += ll_constant
                - .5 * (logdet_sigma0 + Ydata.n_cols * lambda_cov);
            means[reg_params_i] -= ll;
            stderrs[reg_params_i] += ll * ll;
          }
        } // loop over lambda_bC
      } // loop over cross validation folds
      size_t best_i(0);
      double best_score(- std::numeric_limits<double>::infinity());
      for (size_t i(0); i < means.size(); ++i) {
        means[i] /= nfolds;
        stderrs[i] /= nfolds;
        stderrs[i] -= means[i] * means[i];
        stderrs[i] /= std::sqrt(nfolds);
        switch (params_ptr->cv_score_type) {
        case 0:
          if (means[i] < best_score) {
            best_i = i;
            best_score = means[i];
          }
          break;
        case 1:
          if (means[i] - stderrs[i] < best_score) {
            best_i = i;
            best_score = means[i];
          }
          break;
        default:
          assert(false);
        }
      }
      typedef std::map<std::pair<double,double>, size_t> lambdas_index_type;
      foreach(const lambdas_index_type::value_type& t, lambdas_index) {
        if (t.second == best_i) {
          vec best_lambda(2);
          best_lambda[0] = t.first.first;
          best_lambda[1] = t.first.second;
          return best_lambda;
        }
      }
      std::cerr << "gcf_learn_crf_factor_cv_functor::operator() failed!"
                << std::endl; // Should never happen.
      return vec();
    } // operator()

  }; // namespace impl


  template <>
  gaussian_crf_factor*
  learn_crf_factor_cv<gaussian_crf_factor>
  (std::vector<gaussian_crf_factor::regularization_type>& reg_params,
   vec& means, vec& stderrs,
   const crossval_parameters& cv_params,
   const dataset<gaussian_crf_factor::la_type>& ds, const vector_domain& Y_,
   copy_ptr<vector_domain> X_ptr_,
   const gaussian_crf_factor::parameters& params, unsigned random_seed) {

    assert(params.valid());
    assert(cv_params.valid());

    boost::mt11213b rng(random_seed);
    boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());

    std::vector<vec> lambdas;
    impl::gcf_learn_crf_factor_cv_functor cvfunctor(ds, Y_, X_ptr_, params);
    vec best_lambda =
      crossval_zoom<impl::gcf_learn_crf_factor_cv_functor>
      (lambdas, means, stderrs, cv_params, cvfunctor, unif_int(rng));
    assert(best_lambda.size() ==
           gaussian_crf_factor::regularization_type::nlambdas);

    reg_params.clear();
    gaussian_crf_factor::regularization_type reg;
    reg.regularization = params.reg.regularization;
    foreach(const vec& v, lambdas) {
      reg.lambdas = v;
      reg_params.push_back(reg);
    }
    gaussian_crf_factor::parameters tmp_params(params);
    tmp_params.reg.lambdas = best_lambda;
    return learn_crf_factor<gaussian_crf_factor>(ds, Y_, X_ptr_,
                                                 tmp_params, unif_int(rng));

  } // learn_crf_factor_cv<gaussian_crf_factor>()


  GEN_LEARN_CRF_FACTOR_CV_HYBRID_DEF(gaussian_crf_factor)

}; // namespace sill

#include <sill/macros_undef.hpp>
