#include <limits>

#include <sill/learning/crossval_methods.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/record_conversions.hpp>
#include <sill/learning/discriminative/multiclass_logistic_regression.hpp>
#include <sill/math/permutations.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Protected methods
  //==========================================================================

  void multiclass_logistic_regression::build() {
    // Process dataset and parameters
    assert(params.valid());
    if (ds.num_finite() > 1) {
      finite_offset.push_back(0);
      for (size_t j = 0; j < ds.num_finite(); ++j)
        if (j != label_index_) {
          finite_indices.push_back(j);
          finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
        }
      finite_offset.pop_back();
    }
    if (ds.num_vector() > 0) {
      vector_offset.push_back(0);
      for (size_t j = 0; j < ds.num_vector() - 1; ++j)
        vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
    }
    fixed_record = true;
    lambda = params.lambda;
    if (params.regularization == 1 || params.regularization == 2) {
      if (lambda < 0) {
        std::cerr << "multiclass_logistic_regression was given lambda < 0: "
                  << "lambda = " << lambda << std::endl;
        assert(false);
        return;
      }
    }
    rng.seed(static_cast<unsigned>(params.random_seed));
    for (size_t i(0); i < ds.size(); ++i)
      total_train_weight += ds.weight(i);

    // Initialize weights
    if (params.perturb_init > 0) {
      boost::uniform_real<double> uniform_dist;
      uniform_dist = boost::uniform_real<double>
        (-1 * params.perturb_init, params.perturb_init);
      for (size_t i = 0; i < weights_.f.size1(); ++i)
        for (size_t j = 0; j < weights_.f.size2(); ++j)
          weights_.f(i,j) = uniform_dist(rng);
      for (size_t i = 0; i < weights_.v.size1(); ++i)
        for (size_t j = 0; j < weights_.v.size2(); ++j)
          weights_.v(i,j) = uniform_dist(rng);
      foreach(double& v, weights_.b)
        v = uniform_dist(rng);
    }

    // Initialize for whatever learning algorithm
    switch(params.method) {
    case 0: // Batch gradient descent
      {
        obj_functor_ptr = new objective_functor(*this);
        grad_functor_ptr = new gradient_functor(*this);
        gradient_descent_parameters ga_params;
        ga_params.convergence_zero = params.convergence_zero;
        gradient_method_ptr =
          new gradient_descent_type(*obj_functor_ptr, *grad_functor_ptr,
                                   weights_, ga_params);
      }
      break;
    case 1: // Batch conjugate gradient
      {
        obj_functor_ptr = new objective_functor(*this);
        grad_functor_ptr = new gradient_functor(*this);
        conjugate_gradient_parameters cg_params;
        cg_params.convergence_zero = params.convergence_zero;
        gradient_method_ptr =
          new conjugate_gradient_type(*obj_functor_ptr, *grad_functor_ptr,
                                      weights_, cg_params);
      }
      break;
    case 2: // Stochastic gradient descent
      {
        gradient_ptr =
          new opt_variables(opt_variables::size_type
                            (weights_.f.size1(), weights_.f.size2(),
                             weights_.v.size1(), weights_.v.size2(),
                             weights_.b.size()), 0);
        eta = params.eta;
      }
      break;
    case 3: // Batch conjugate gradient with a diagonal preconditioner
      {
        obj_functor_ptr = new objective_functor(*this);
        grad_functor_ptr = new gradient_functor(*this);
        prec_functor_ptr = new preconditioner_functor(*this);
        conjugate_gradient_parameters cg_params;
        cg_params.convergence_zero = params.convergence_zero;
        gradient_method_ptr =
          new prec_conjugate_gradient_type(*obj_functor_ptr, *grad_functor_ptr,
					   *prec_functor_ptr, weights_,
					   cg_params);
      }
      break;
    default:
      assert(false);
    }

    while (iteration_ < params.init_iterations) {
      if (!step()) {
        if (params.debug > 1)
          std::cerr << "multiclass_logistic_regression terminated"
                    << " optimization after " << iteration_ << "iterations."
                    << std::endl;
        break;
      }
    }
    fixed_record = false;
    if (iteration_ == params.init_iterations && params.init_iterations > 0)
      if (params.debug > 0)
        std::cerr << "WARNING: multiclass_logistic_regression terminated"
                  << " optimization after init_iterations="
                  << params.init_iterations << " steps! Consider doing more."
                  << std::endl;
  } // end of function build()

  void multiclass_logistic_regression::finish_probabilities(vec& v) const {
    if (params.resolve_numerical_problems) {
      for (size_t k(0); k < nclasses_; ++k) {
        if (v(k) > log_max_double) {
          double maxval(v(max_index(v, rng)));
          for (size_t k(0); k < nclasses_; ++k) {
            if (v(k) == maxval)
              v(k) = 1;
            else
              v(k) = 0;
          }
          v /= sum(v);
          return;
        }
      }
      for (size_t k(0); k < nclasses_; ++k)
        v(k) = exp(v(k));
      double tmpsum(sum(v));
      if (tmpsum == 0) {
        v.ones();
        v /= nclasses_;
        return;
      }
      v /= tmpsum;
    } else {
      for (size_t k(0); k < nclasses_; ++k) {
        if (v(k) > log_max_double) {
          throw std::runtime_error("multiclass_logistic_regression had overflow when computing probabilities.  To deal with such overflows in a hacky (but reasonable) way, use the parameter resolve_numerical_problems.");
        }
        v(k) = exp(v(k));
      }
      if (sum(v) == 0)
        throw std::runtime_error("multiclass_logistic_regression got all zeros when computing probabilities.  To deal with such issues in a hacky (but reasonable) way, use the parameter resolve_numerical_problems.");
      v /= sum(v);
    }
  }

  void
  multiclass_logistic_regression::my_probabilities(const record& example,
                                                   vec& v, const mat& w_fin_,
                                                   const mat& w_vec_,
                                                   const vec& b_) const {
    v = b_;
    const std::vector<size_t>& findata = example.finite();
    for (size_t k(0); k < nclasses_; ++k) {
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        v(k) += w_fin_(k, finite_offset[j] + val);
      }
    }
    if (w_vec_.size() != 0)
      v += w_vec_ * example.vector();
    finish_probabilities(v);
  }

  void
  multiclass_logistic_regression::my_probabilities(const assignment& example,
                                                   vec& v, const mat& w_fin_,
                                                   const mat& w_vec_,
                                                   const vec& b_) const {
    v = b_;
    const finite_assignment& fa = example.finite();
    const vector_assignment& va = example.vector();
    for (size_t k(0); k < nclasses_; ++k) {
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(safe_get(fa, finite_seq[finite_indices[j]]));
        v(k) += w_fin_(k, finite_offset[j] + val);
      }
      for (size_t j(0); j < vector_seq.size(); ++j) {
        const vec& vecdata = safe_get(va, vector_seq[j]);
        for (size_t j2(0); j2 < vector_seq[j]->size(); ++j2) {
          size_t ind(vector_offset[j] + j2);
          v(k) += w_vec_(k,ind) * vecdata[j2];
        }
      }
    }
    finish_probabilities(v);
  }

  void
  multiclass_logistic_regression::my_probabilities(const record& example,
                                                   vec& v) const {
    if (fixed_record ||
        ((finite_offset.size() == 0 ||
          example.finite_numbering_ptr->size() == finite_offset.size() + 1) &&
         (vector_offset.size() == 0 ||
          example.vector_numbering_ptr->size() == vector_offset.size())))
      return my_probabilities(example, v, weights_.f, weights_.v, weights_.b);
    else
      return my_probabilities(example.assignment(), v, weights_.f, weights_.v,
                              weights_.b);
  }

  void
  multiclass_logistic_regression::my_probabilities(const assignment& example,
                                                   vec& v) const {
    return my_probabilities(example, v, weights_.f, weights_.v, weights_.b);
  }

  void
  multiclass_logistic_regression::
  add_raw_gradient(opt_variables& gradient, double& acc, double& ll,
                   const record& example, double weight, double alt_weight,
                   const opt_variables& x) const {
    vec v;
    my_probabilities(example, v, x.f, x.v, x.b);
    const std::vector<size_t>& findata = example.finite();
    const vec& vecdata = example.vector();
    size_t label_val(findata[label_index_]);
    size_t pred_(max_index(v, rng));
    if (label_val == pred_)
      acc += weight;
    ll += weight * std::log(v[label_val]);
    // Update gradients
    v(label_val) -= 1;
    v *= weight * alt_weight;
    for (size_t j(0); j < finite_indices.size(); ++j) {
      size_t val(finite_offset[j] + findata[finite_indices[j]]);
      for (size_t k(0); k < nclasses_; ++k) {
        gradient.f(k, val) += v(k);
      }
    }
    if (vecdata.size() != 0)
      gradient.v += outer_product(v, vecdata);
    gradient.b += v;
  } // add_raw_gradient()

  void
  multiclass_logistic_regression::
  add_reg_gradient(opt_variables& gradient, double alt_weight,
                   const opt_variables& x) const {
    double w(alt_weight * lambda);
    switch (params.regularization) {
    case 0: // none
      break;
    case 1: // L-1
      // TODO: Figure out a better way to do this.
      //       (Add more functionality to vector.hpp and matrix.hpp?)
      for (size_t i(0); i < nclasses_; ++i) {
        for (size_t j(0); j < x.f.size2(); ++j) {
          if (x.f(i,j) > 0)
            gradient.f(i,j) += w;
          else if (x.f(i,j) < 0)
            gradient.f(i,j) -= w;
        }
        for (size_t j(0); j < x.v.size2(); ++j) {
          if (x.v(i,j) > 0)
            gradient.v(i,j) += w;
          else if (x.v(i,j) < 0)
            gradient.v(i,j) -= w;
        }
        if (x.b(i) > 0)
          gradient.b(i) += w;
        else if (x.b(i) < 0)
          gradient.b(i) -= w;
      }
      break;
    case 2: // L-2
      gradient.f += w * x.f;
      gradient.v += w * x.v;
      gradient.b += w * x.b;
      break;
    default:
      assert(false);
    }
  } // add_reg_gradient()

  std::pair<double,double>
  multiclass_logistic_regression::my_gradient(opt_variables& gradient,
                                              const opt_variables& x) const {
    double train_acc(0.);
    double train_log_like(0.);
    gradient.zeros();
    dataset::record_iterator it_end(ds.end());
    size_t i(0); // index into dataset
    for (dataset::record_iterator it(ds.begin()); it != it_end; ++it) {
      // Compute v = prediction for *it.  Update accuracy, log likelihood.
      add_raw_gradient
        (gradient, train_acc, train_log_like, *it, ds.weight(i), 1, x);
      ++i;
    }
    gradient.f /= total_train_weight;
    gradient.v /= total_train_weight;
    gradient.b /= total_train_weight;

    // Update gradients to account for regularization
    add_reg_gradient(gradient, 1, x);
    return std::make_pair(train_acc, train_log_like);
  } // end of function my_gradient()

  void
  multiclass_logistic_regression::my_hessian_diag
  (opt_variables& hd, const opt_variables& x) const {

    if (hd.size() != x.size())
      hd.resize(x.size());
    hd.zeros();
    dataset::record_iterator it_end(ds.end());
    size_t i(0); // index into ds
    vec v;
    vec vecdata;
    for (dataset::record_iterator it(ds.begin()); it != it_end; ++it) {
      my_probabilities(*it, v, x.f, x.v, x.b);
      v -= elem_mult(v, v);
      v *= ds.weight(i);
      const std::vector<size_t>& findata = (*it).finite();
      for (size_t j(0); j < finite_indices.size(); ++j) {
	size_t val(findata[finite_indices[j]]);
	hd.f.add_column(finite_offset[j] + val, v);
      }
      elem_mult_out((*it).vector(), (*it).vector(), vecdata);
      hd.v += outer_product(v, vecdata);
      hd.b += v;
      ++i;
    }
    hd.f /= total_train_weight;
    hd.v /= total_train_weight;
    hd.b /= total_train_weight;

    switch (params.regularization) {
    case 0:
      break;
    case 1:
      // This is supposed to be 0 (even at the discontinuity), right?
      break;
    case 2:
      hd.f += lambda;
      hd.v += lambda;
      hd.b += lambda;
      break;
    default:
      assert(false);
    }

  } // my_hessian_diag()

  bool multiclass_logistic_regression::step_gradient_method() {
    if (!gradient_method_ptr)
      return false;

    double prev_train_log_like(train_log_like);
    if (!gradient_method_ptr->step())
      return false;
    train_log_like = - gradient_method_ptr->objective();
    if (train_log_like < prev_train_log_like) {
      if (params.debug > 0)
        std::cerr << "multiclass_logistic_regression took a step which "
                  << "lowered the regularized log likelihood from "
                  << prev_train_log_like << " to " << train_log_like
                  << "; last eta = " << eta
                  << std::endl;
    }
    if (params.debug > 1) {
      std::cerr << "change in regularized log likelihood = "
                << (train_log_like - prev_train_log_like) << std::endl;
    }
    // Check for convergence
    if (fabs(train_log_like - prev_train_log_like) < params.convergence_zero) {
      if (params.debug > 1)
        std::cerr << "multiclass_logistic_regression converged:"
                  << " regularized training log likelihood changed from "
                  << prev_train_log_like << " to " << train_log_like
                  << "; exiting early (iteration " << iteration() << ")."
                  << std::endl;
      return false;
    }

    ++iteration_;
    return true;
  }

  bool multiclass_logistic_regression::step_stochastic_gradient_descent() {
    if (!gradient_ptr)
      return false;
    if (!(o.next()))
      return false;
    const record& r = o.current();
    double r_weight(o.weight());
    total_train_weight += r_weight;
    // Compute v = prediction for *it.  Update accuracy, log likelihood.
    gradient_ptr->f.zeros_memset();
    gradient_ptr->v.zeros_memset();
    gradient_ptr->b.zeros_memset();
    add_raw_gradient(*gradient_ptr, train_acc, train_log_like, r, r_weight,
                     eta, weights_);
    // Update gradients to account for regularization
    add_reg_gradient(*gradient_ptr, eta, weights_);

    if (params.debug > 1) {
      std::cerr << "Gradient extrema info:\n";
      gradient_ptr->print_extrema_info(std::cerr);
    }

    // Update weights and learning rate eta.
    //  (Note that the gradient has already been multiplied by eta.)
    weights_.f -= gradient_ptr->f;
    weights_.v -= gradient_ptr->v;
    weights_.b -= gradient_ptr->b;
    eta *= params.mu;

    if (params.debug > 1) {
      std::cerr << "Weights extrema info:\n";
      weights_.print_extrema_info(std::cerr);
    }

    ++iteration_;
    return true;
  } // end of function: void step_stochastic_gradient_descent()

  vec multiclass_logistic_regression::choose_lambda_helper::operator()
  (vec& means, vec& stderrs, const std::vector<vec>& lambdas, size_t nfolds,
   unsigned random_seed) const {
    assert(lambdas.size() > 0);
    assert(nfolds > 1 && nfolds <= ds_ptr->size());
    for (size_t j(0); j < lambdas.size(); ++j) {
      assert(lambdas[j].size() == 1);
    }
    means.resize(lambdas.size());
    means.zeros_memset();
    stderrs.resize(lambdas.size());
    stderrs.zeros_memset();
    multiclass_logistic_regression_parameters params(params_);
    boost::mt11213b rng(random_seed);
    boost::uniform_int<int> unif_int(0, std::numeric_limits<int>::max());
    dataset_view permuted_view(*ds_ptr);
    permuted_view.set_record_indices(randperm(ds_ptr->size(), rng));
    dataset_view fold_train(permuted_view);
    dataset_view fold_test(permuted_view);
    fold_train.save_record_view();
    fold_test.save_record_view();
    for (size_t fold(0); fold < nfolds; ++fold) {
      if (fold != 0) {
        fold_train.restore_record_view();
        fold_test.restore_record_view();
      }
      fold_train.set_cross_validation_fold(fold, nfolds, false);
      fold_test.set_cross_validation_fold(fold, nfolds, true);
      for (size_t j(0); j < lambdas.size(); ++j) {
        params.lambda = lambdas[j][0];
        dataset_statistics stats(fold_train);
        multiclass_logistic_regression mlr(stats, params);
        size_t i(0);
        double val(0);
        foreach(const record& r, fold_test.records()) {
          val -= std::log(mlr.probabilities(r)[mlr.label(fold_test,i)]);
          ++i;
        }
        val /= fold_test.size();
        means[j] += val;
        stderrs[j] += val * val;
      }
    }

    for (size_t k(0); k < lambdas.size(); ++k) {
      means[k] /= nfolds;
      stderrs[k] /= nfolds;
      stderrs[k] = sqrt((stderrs[k] - means[k] * means[k]) / nfolds);
    }
    size_t min_i(min_index(means, rng));
    return lambdas[min_i];
  } // end of choose_lambda_helper::operator()

  // Getters and helpers
  //==========================================================================

  double multiclass_logistic_regression::train_accuracy() const {
    switch(params.method) {
    case 0:
    case 1:
      return train_acc;
    case 2:
      return (total_train_weight == 0 ? -1 : train_acc / total_train_weight);
    default:
      assert(false);
      return -1;
    }
  }

  void multiclass_logistic_regression::fix_record(const record& r) {
    // Set finite_indices, finite_offset, vector_offset.
    // Set finite_seq, vector_seq?
    assert(false); // TO DO
    fixed_record = true;
  }

  void multiclass_logistic_regression::unfix_record() {
    fixed_record = false;
  }

  void multiclass_logistic_regression::add_gradient(opt_variables& grad,
                                                    const assignment& a,
                                                    double w) const {
    const finite_assignment& fa = a.finite();
    const vector_assignment& va = a.vector();
    size_t label_val(safe_get(fa,label_));
    for (size_t j(0); j < finite_indices.size(); ++j) {
      size_t val(safe_get(fa, finite_seq[finite_indices[j]]));
      grad.f(label_val, finite_offset[j] + val) -= w;
    }
    for (size_t j(0); j < vector_seq.size(); ++j) {
      const vec& vecdata = safe_get(va, vector_seq[j]);
      for (size_t j2(0); j2 < vecdata.size(); ++j2) {
        size_t ind(vector_offset[j] + j2);
        grad.v(label_val, ind) -= vecdata[j2] * w;
      }
    }
    grad.b(label_val) -= w;
  }

  void multiclass_logistic_regression::add_gradient(opt_variables& grad,
                                                    const record& r,
                                                    double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      add_gradient(grad, r.assignment(), w);
    } else {
      const std::vector<size_t>& findata = r.finite();
      size_t label_val(findata[label_index_]);
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        grad.f(label_val, finite_offset[j] + val) -= w;
      }
      grad.v.subtract_row(label_val, r.vector() * w);
      grad.b(label_val) -= w;
    }
  }

  void
  multiclass_logistic_regression::add_expected_gradient
  (opt_variables& grad, const assignment& a, const table_factor& fy,
   double w) const {
    // Get marginal over label variable.
    table_factor
      label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
    finite_assignment tmpa;
    vec r_vector(grad.v.size2());
    vector_assignment2vector(a.vector(), vector_seq, r_vector);
    for (size_t label_val(0); label_val < nclasses_; ++label_val) {
      tmpa[label_] = label_val;
      double label_prob(label_marginal(tmpa));
      if (label_prob == 0)
        continue;
      label_prob *= w;
      grad.v.subtract_row(label_val, label_prob * r_vector);
      grad.b(label_val) -= label_prob;
    }
    tmpa = a.finite();
    foreach(const finite_assignment& fa, assignments(fy.arguments())) {
      finite_assignment::const_iterator fa_end(fa.end());
      for (finite_assignment::const_iterator fa_it(fa.begin());
           fa_it != fa_end; ++fa_it)
        tmpa[fa_it->first] = fa_it->second;
      size_t label_val(tmpa[label_]);
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(tmpa[finite_seq[finite_indices[j]]]);
        grad.f(label_val, finite_offset[j] + val) -= w * fy(fa);
      }
    }
  }

  void
  multiclass_logistic_regression::add_expected_gradient
  (opt_variables& grad, const record& r, const table_factor& fy,
   double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      add_expected_gradient(grad, r.assignment(), fy, w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      for (size_t label_val(0); label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        label_prob *= w;
        grad.v.subtract_row(label_val, label_prob * r.vector());
        grad.b(label_val) -= label_prob;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j(0); j < finite_indices.size(); ++j) {
          size_t val(tmpa[finite_seq[finite_indices[j]]]);
          grad.f(label_val, finite_offset[j] + val) -= w * fy(fa);
        }
      }
    }
  }

  void
  multiclass_logistic_regression::add_combined_gradient
  (opt_variables& grad, const record& r, const table_factor& fy,
   double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      assignment tmpa(r.assignment());
      add_gradient(grad, tmpa, w);
      add_expected_gradient(grad, tmpa, fy, -w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      const std::vector<size_t>& findata = r.finite();
      for (size_t label_val(0); label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        if (label_val == findata[label_index_])
          label_prob -= 1.;
        label_prob *= -w;
        grad.v.subtract_row(label_val, label_prob * r.vector());
        grad.b(label_val) -= label_prob;
      }
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        grad.f(findata[label_index_], finite_offset[j] + val) -= w;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j(0); j < finite_indices.size(); ++j) {
          size_t val(tmpa[finite_seq[finite_indices[j]]]);
          grad.f(label_val, finite_offset[j] + val) += w * fy(fa);
        }
      }
    }
  }

  void multiclass_logistic_regression::add_expected_squared_gradient
  (opt_variables& hd, const assignment& a, const table_factor& fy,
   double w) const {
    assert(false); // TO DO
  }

  void multiclass_logistic_regression::add_expected_squared_gradient
  (opt_variables& hd, const record& r, const table_factor& fy, double w) const {
    if ((finite_offset.size() != 0 &&
         r.finite_numbering_ptr->size() != finite_offset.size()) ||
        (vector_offset.size() != 0 &&
         r.vector_numbering_ptr->size() != vector_offset.size())) {
      add_expected_squared_gradient(hd, r.assignment(), fy, w);
    } else {
      // Get marginal over label variable.
      table_factor
        label_marginal(fy.marginal(make_domain<finite_variable>(label_)));
      finite_assignment tmpa;
      for (size_t label_val(0); label_val < nclasses_; ++label_val) {
        tmpa[label_] = label_val;
        double label_prob(label_marginal(tmpa));
        if (label_prob == 0)
          continue;
        label_prob *= w;
        hd.v.subtract_row(label_val,
                          label_prob * elem_mult(r.vector(),r.vector()));
        hd.b(label_val) -= label_prob;
      }
      tmpa = r.finite_assignment();
      foreach(const finite_assignment& fa, assignments(fy.arguments())) {
        finite_assignment::const_iterator fa_end(fa.end());
        for (finite_assignment::const_iterator fa_it(fa.begin());
             fa_it != fa_end; ++fa_it)
          tmpa[fa_it->first] = fa_it->second;
        size_t label_val(tmpa[label_]);
        for (size_t j(0); j < finite_indices.size(); ++j) {
          size_t val(tmpa[finite_seq[finite_indices[j]]]);
          hd.f(label_val, finite_offset[j] + val) -= w * fy(fa);
        }
      }
    }
  }

  // Methods for iterative learners
  //==========================================================================

    bool multiclass_logistic_regression::step() {
      switch(params.method) {
      case 0:
      case 1:
      case 3:
        return step_gradient_method();
      case 2:
        return step_stochastic_gradient_descent();
      default:
        assert(false);
        return false;
      }
    }

  // Save and load methods
  //==========================================================================

  void multiclass_logistic_regression::save(std::ofstream& out,
                                            size_t save_part,
                                            bool save_name) const {
    assert(false);
    // TO DO: SAVE POINTERS TO STUFF, ETC.
    base::save(out, save_part, save_name);
    params.save(out);
    out << eta << " " << train_acc << " " << train_log_like
        << " " << iteration_ << " " << total_train_weight << "\n";
    for (size_t i = 0; i < weights_.f.size1(); ++i)
      out << weights_.f.row(i) << " ";
    out << "\n";
    for (size_t i = 0; i < weights_.f.size1(); ++i)
      out << weights_.v.row(i) << " ";
    out << "\n" << weights_.b << "\n";
  }

  bool multiclass_logistic_regression::load(std::ifstream& in,
                                            const datasource& ds,
                                            size_t load_part) {
    assert(false);
    // TO DO: CLEAR POINTERS TO STUFF, ETC.
    if (!(base::load(in, ds, load_part)))
      return false;
    finite_seq = ds.finite_list();
    vector_seq = ds.vector_list();
    finite_offset.clear();
    finite_indices.clear();
    if (ds.num_finite() > 1) {
      finite_offset.push_back(0);
      for (size_t j = 0; j < ds.num_finite(); ++j)
        if (j != label_index_) {
          finite_indices.push_back(j);
          finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
        }
      finite_offset.pop_back();
    }
    vector_offset.clear();
    if (ds.num_vector() > 0) {
      vector_offset.push_back(0);
      for (size_t j = 0; j < ds.num_vector() - 1; ++j)
        vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
    }
    params.load(in);
    rng.seed(static_cast<unsigned>(params.random_seed));
    lambda = params.lambda;
    nclasses_ = nclasses();
    std::string line;
    getline(in, line);
    std::istringstream is(line);
    if (!(is >> eta))
      assert(false);
    assert(eta > 0 && eta <= 1);
    if (!(is >> train_acc))
      assert(false);
    if (!(is >> train_log_like))
      assert(false);
    if (!(is >> iteration_))
      assert(false);
    if (!(is >> total_train_weight))
      assert(false);
    getline(in, line);
    is.clear();
    is.str(line);
    weights_.f.resize(nclasses_, ds.finite_dim() - nclasses_);
    weights_.v.resize(nclasses_, ds.vector_dim());
    for (size_t j = 0; j < nclasses_; ++j) {
      read_vec(is, tmpvec);
      weights_.f.set_row(j, tmpvec);
    }
    getline(in, line);
    is.clear();
    is.str(line);
    for (size_t j = 0; j < nclasses_; ++j) {
      read_vec(is, tmpvec);
      weights_.v.set_row(j, tmpvec);
    }
    getline(in, line);
    is.clear();
    is.str(line);
    read_vec(is, weights_.b);
    return true;
  }

  // Methods for choosing lambda via cross validation
  //==========================================================================

  double multiclass_logistic_regression::choose_lambda
  (std::vector<vec>& lambdas, vec& means, vec& stderrs,
   const crossval_parameters& cv_params, boost::shared_ptr<dataset> ds_ptr,
   const multiclass_logistic_regression_parameters& params,
   unsigned random_seed) {
    choose_lambda_helper clh(ds_ptr, params);
    vec best_lambda =
      crossval_zoom<choose_lambda_helper>
      (lambdas, means, stderrs, cv_params, clh, random_seed);
    assert(best_lambda.size() == 1);
    return best_lambda[0];
  }

} // namespace sill

#include <sill/macros_undef.hpp>
