#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/log_reg_crf_factor.hpp>
#include <sill/factor/table_crf_factor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/crf/pwl_crf_learner.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/syn_oracle_bayes_net.hpp>
#include <sill/learning/dataset/syn_oracle_majority.hpp>
#include <sill/learning/dataset/assignment_dataset.hpp>
#include <sill/model/model_products.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file pwl_crf_learner_test.cpp  Test the piecewise likelihood-based CRF
 *                                 learner.
 */

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Dataset parameters
  size_t nsamples = 500;
  size_t n = 5; // length of chains P(X) and P(Y|X)
  unsigned model_seed = 3078574;
  unsigned oracle_seed = 1284392;
  // CRF learning parameters
  //  unsigned random_seed = 6937111;
  bool use_count_regressors = true; // use count_regressor when applicable
  bool use_per_variable_inputs = true; // input maps are from single Y's to X
  multiclass_logistic_regression_parameters mlr_params;
  mlr_params.init_iterations = 100;
  mlr_params.regularization = 0;
  mlr_params.lambda = .01;
  mlr_params.method = 1;
  mlr_params.convergence_zero = .01;
  bool learn_tree = true;
  size_t debug_mode = 2;
  double edge_reg = 0;
  size_t score_type = 1;
//  bool print_true_scores = false;

  // Create a model
  universe u;
  boost::mt11213b rng(oracle_seed);
  decomposable<table_factor> YXmodel;
  crf_model<table_crf_factor> YgivenXmodel;
  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
    Y_X_and_map(create_random_chain_crf(YXmodel, YgivenXmodel, n, u,
                                        model_seed));
  model_product_inplace(YgivenXmodel, YXmodel);
  finite_var_vector Y(Y_X_and_map.get<0>());
  finite_var_vector X(Y_X_and_map.get<1>());
  finite_var_vector YX(Y);
  YX.insert(YX.end(), X.begin(), X.end());
  std::map<finite_variable*, copy_ptr<finite_domain> >
    Y2X_map(Y_X_and_map.get<2>());
  cout << "True model for P(Y,X):\n" << YXmodel << "\n" << endl;
//  cout << "True model for P(Y|X):\n" << YgivenXmodel << "\n" << endl;

  // Generate a dataset
  cout << "Sampling " << nsamples << " training samples from the model" << endl;
  assignment_dataset ds(YX, vector_var_vector(),
                        std::vector<variable::variable_typenames>
                        (YX.size(), variable::FINITE_VARIABLE));
  for (size_t i(0); i < nsamples; ++i) {
    finite_assignment fa(YXmodel.sample(rng));
    ds.insert(assignment(fa));
  }

  // Learn a model
  cout << "Learning CRFs using Y = " << Y << "\n"
       << " and X = " << X << endl;

  cout << "\nLearning a CRF P(Y|X) using the pwl_crf_learner\n"
       << endl;

  finite_domain Yset;
  Yset.insert(Y.begin(), Y.end());

  if (use_count_regressors) {
    pwl_crf_learner<table_crf_factor>::parameters pwlcl_params;
    pwlcl_params.score_type = score_type;
    pwlcl_params.learn_tree = learn_tree;
    pwlcl_params.edge_reg = edge_reg;
    boost::shared_ptr<table_crf_factor::parameters>
      tcf_params_ptr(new table_crf_factor::parameters());
    tcf_params_ptr->reg.lambdas[0] = 1. / ds.size();
    pwlcl_params.crf_factor_params_ptr = tcf_params_ptr;
    pwlcl_params.DEBUG = debug_mode;

    if (use_per_variable_inputs) {
      pwl_crf_learner<table_crf_factor>
        pwlcl_learner(ds, Yset, Y2X_map, pwlcl_params);
      cout << "Learned CRF structure:\n" << pwlcl_learner.current_graph()
           << endl;
      crf_model<table_crf_factor> model(pwlcl_learner.current_model());
      double ll(0);
      foreach(const assignment& a, ds.assignments())
        ll += model.log_likelihood(a.finite());
      ll /= ds.size();
      cout << "Learned CRF model's average data log likelihood: " << ll << endl;
    } else {
      std::map<finite_domain, copy_ptr<finite_domain> > pwlcl_X_map;
      for (size_t i(0); i < Y.size() - 1; ++i) {
        for (size_t j(i+1); j < Y.size(); ++j) {
          copy_ptr<finite_domain> tmpdom;
          tmpdom->insert(Y2X_map[Y[i]]->begin(), Y2X_map[Y[i]]->end());
          tmpdom->insert(Y2X_map[Y[j]]->begin(), Y2X_map[Y[j]]->end());
          pwlcl_X_map[make_domain<finite_variable>(Y[i], Y[j])] = tmpdom;
        }
      }
      for (size_t i(0); i < Y.size(); ++i) {
        pwlcl_X_map[make_domain<finite_variable>(Y[i])] = Y2X_map[Y[i]];;
      }
      pwl_crf_learner<table_crf_factor>
        pwlcl_learner(ds, Yset, pwlcl_X_map, pwlcl_params);
      cout << "Learned CRF structure:\n" << pwlcl_learner.current_graph()
           << endl;
      /*
        if (print_true_scores)
        pwlcl_learner.debug_truth(YXmodel, YgivenXmodel);
      */
      crf_model<table_crf_factor> model(pwlcl_learner.current_model());
      double ll(0);
      foreach(const assignment& a, ds.assignments())
        ll += model.log_likelihood(a.finite());
      ll /= ds.size();
      cout << "Learned CRF model's average data log likelihood: " << ll << endl;
    }
  } else {
    pwl_crf_learner<log_reg_crf_factor>::parameters pwlcl_params;
    pwlcl_params.score_type = score_type;
    pwlcl_params.learn_tree = learn_tree;
    pwlcl_params.edge_reg = edge_reg;
    boost::shared_ptr<log_reg_crf_factor::parameters>
      lrcf_params_ptr(new log_reg_crf_factor::parameters(mlr_params, u));
    pwlcl_params.crf_factor_params_ptr = lrcf_params_ptr;
    pwlcl_params.DEBUG = debug_mode;

    if (use_per_variable_inputs) {
      std::map<finite_variable*, copy_ptr<domain> > Y2X_finite_map;
      foreach(finite_variable* fv, Y) {
        copy_ptr<domain> tmpdom;
        foreach(finite_variable* tmpfv, *(Y2X_map[fv]))
          tmpdom->insert(tmpfv);
        Y2X_finite_map[fv] = tmpdom;
      }
      pwl_crf_learner<log_reg_crf_factor>
        pwlcl_learner(ds, Yset, Y2X_finite_map, pwlcl_params);
      cout << "Learned CRF structure:\n" << pwlcl_learner.current_graph()
           << endl;
      crf_model<log_reg_crf_factor> model(pwlcl_learner.current_model());
      double ll(0);
      foreach(const assignment& a, ds.assignments())
        ll += model.log_likelihood(a.finite());
      ll /= ds.size();
      cout << "Learned CRF model's average data log likelihood: " << ll << endl;
    } else {
      std::map<finite_domain, copy_ptr<domain> > pwlcl_X_map;
      for (size_t i(0); i < Y.size() - 1; ++i) {
        for (size_t j(i+1); j < Y.size(); ++j) {
          copy_ptr<domain> tmpdom;
          tmpdom->insert(Y2X_map[Y[i]]->begin(), Y2X_map[Y[i]]->end());
          tmpdom->insert(Y2X_map[Y[j]]->begin(), Y2X_map[Y[j]]->end());
          pwlcl_X_map[make_domain<finite_variable>(Y[i], Y[j])] = tmpdom;
        }
      }
      for (size_t i(0); i < Y.size(); ++i) {
        copy_ptr<domain> tmpdom;
        tmpdom->insert(Y2X_map[Y[i]]->begin(), Y2X_map[Y[i]]->end());
        pwlcl_X_map[make_domain<finite_variable>(Y[i])] = tmpdom;
      }
      pwl_crf_learner<log_reg_crf_factor>
        pwlcl_learner(ds, Yset, pwlcl_X_map, pwlcl_params);
      cout << "Learned CRF structure:\n" << pwlcl_learner.current_graph()
           << endl;
      /*
        if (print_true_scores)
        pwlcl_learner.debug_truth(YXmodel, YgivenXmodel);
      */
      crf_model<log_reg_crf_factor> model(pwlcl_learner.current_model());
      double ll(0);
      foreach(const assignment& a, ds.assignments())
        ll += model.log_likelihood(a.finite());
      ll /= ds.size();
      cout << "Learned CRF model's average data log likelihood: " << ll << endl;
    }
  }

  return 0;
}
