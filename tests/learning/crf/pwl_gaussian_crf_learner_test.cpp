#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/learning/crf/pwl_crf_learner.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file pwl_gaussian_crf_learner_test.cpp  Test the PWL-based CRF learner.
 */

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Dataset parameters
  size_t nsamples = 500;
  size_t n = 10; // length of chains P(X) and P(Y|X)
  unsigned model_seed = 3078574;
  unsigned oracle_seed = 1284392;
  double b_max = 5;
  double c_max = 2;
  double variance = 1;
  double YYcorrelation = .5;
  double YXcorrelation = .1;
  double XXcorrelation = .1;
  bool add_cross_factors = false;
  // CRF learning parameters
  bool learn_tree = true;
  size_t debug_mode = 2;
  double edge_reg = 0;
  size_t score_type = 2;
  bool cv_log_scale = true;

  // Create a model
  universe u;
  boost::mt11213b rng(oracle_seed);
  decomposable<canonical_gaussian> YXmodel;
  crf_model<gaussian_crf_factor> YgivenXmodel;
  boost::tuple<vector_var_vector, vector_var_vector,
               std::map<vector_variable*, copy_ptr<vector_domain> > >
    Y_X_and_map(create_random_gaussian_crf
                (YXmodel, YgivenXmodel, n, u, "chain", b_max, c_max, variance,
                 YYcorrelation, YXcorrelation, XXcorrelation, add_cross_factors,
                 model_seed));
  vector_var_vector Y(Y_X_and_map.get<0>());
  vector_var_vector X(Y_X_and_map.get<1>());
  vector_var_vector YX(Y);
  YX.insert(YX.end(), X.begin(), X.end());
  std::map<vector_variable*, copy_ptr<vector_domain> >
    Y2X_map(Y_X_and_map.get<2>());
  cout << "True model for P(Y,X):\n" << YXmodel << "\n" << endl;
//  cout << "True model for P(Y|X):\n" << YgivenXmodel << "\n" << endl;

  // Generate a dataset
  cout << "Sampling " << nsamples << " training samples from the model" << endl;
  vector_dataset<> ds(finite_var_vector(), YX, 
                    std::vector<variable::variable_typenames>
                    (YX.size(), variable::VECTOR_VARIABLE));
  for (size_t i(0); i < nsamples; ++i) {
    vector_assignment fa(YXmodel.sample(rng));
    ds.insert(assignment(fa));
  }

  // Learn a model
  cout << "Learning CRFs using Y = " << Y << "\n"
       << " and X = " << X << endl;

  cout << "\nLearning a CRF P(Y|X) using the pwl_crf_learner\n"
       << endl;

  vector_domain Yset;
  Yset.insert(Y.begin(), Y.end());

  pwl_crf_learner<gaussian_crf_factor>::parameters pwlcl_params;
  pwlcl_params.score_type = score_type;
  pwlcl_params.learn_tree = learn_tree;
  pwlcl_params.edge_reg = edge_reg;
  crossval_parameters
    cv_params(gaussian_crf_factor::regularization_type::nlambdas);
  cv_params.minvals = 10.;
  cv_params.maxvals = .001;
  cv_params.nvals = 10;
  cv_params.zoom = 0;
  cv_params.log_scale = cv_log_scale;
  pwlcl_params.cv_params = cv_params;
  pwlcl_params.crf_factor_params_ptr.reset
    (new gaussian_crf_factor::parameters());
  pwlcl_params.DEBUG = debug_mode;

  crf_X_mapping<gaussian_crf_factor> X_mapping(Y2X_map);
  pwl_crf_learner<gaussian_crf_factor>
    pwlcl_learner(ds, Yset, X_mapping, pwlcl_params);
  cout << "Learned CRF structure:\n" << pwlcl_learner.current_graph()
       << endl;
  crf_model<gaussian_crf_factor> model(pwlcl_learner.current_model());
  double ll(0);
  foreach(const assignment& a, ds.assignments())
    ll += model.log_likelihood(a.vector());
  ll /= ds.size();
  cout << "Learned CRF model's average data log likelihood: " << ll << endl;

  return 0;
}
