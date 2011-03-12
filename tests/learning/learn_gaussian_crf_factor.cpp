#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/crf/crf_parameter_learner.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/vector_assignment_dataset.hpp>
#include <sill/learning/learn_crf_factor.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file learn_gaussian_crf_factor.cpp  Test learning conditional Gaussians.
 */

static int usage() {
  std::cout << "usage: ./learn_gaussian_crf_factor "
            << "([ntrain] [ntest] [Ysize] [Xsize])"
            << std::endl;
  return 1;
}

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Dataset parameters
  size_t ntrain = 100;
  size_t ntest = 1000;
  size_t Ysize = 2;
  size_t Xsize = 2;
  unsigned oracle_seed = 1284392;
  double b_max = 5;
  double spread = 2;
  double cov_strength = 1;

  if (argc == 5) {
    std::istringstream is(argv[1]);
    if (!(is >> ntrain))
      return usage();
    is.clear();
    is.str(argv[2]);
    if (!(is >> ntest))
      return usage();
    is.clear();
    is.str(argv[3]);
    if (!(is >> Ysize))
      return usage();
    is.clear();
    is.str(argv[4]);
    if (!(is >> Xsize))
      return usage();
  }

  // Create a conditional Gaussian
  universe u;
  boost::mt11213b rng(oracle_seed);
  boost::uniform_int<unsigned> unif_int(0,std::numeric_limits<unsigned>::max());
  vector_var_vector Y, X;
  for (size_t j(0); j < Ysize; ++j)
    Y.push_back(u.new_vector_variable(1));
  for (size_t j(0); j < Xsize; ++j)
    X.push_back(u.new_vector_variable(1));
  vector_var_vector YX(sill::concat(Y, X));
  moment_gaussian truth(make_marginal_gaussian_factor
                        (sill::concat(Y,X), b_max, spread, cov_strength, rng));
  if (1) {
    canonical_gaussian cg1(truth);
    canonical_gaussian cg2(truth);
    cg2.enforce_psd(truth.mean());
    if (cg1.inf_matrix() != cg2.inf_matrix())
      assert(false);
    if (cg1.inf_vector() != cg2.inf_vector())
      assert(false);
    if (cg1.log_multiplier() != cg2.log_multiplier())
      assert(false);
  }
  moment_gaussian
    true_conditional(truth.conditional(make_domain<vector_variable>(X)));

  // Generate a dataset
  cout << "Sampling " << (ntrain+ntest) << " samples from the model" << endl;
  vector_assignment_dataset<> ds(finite_var_vector(), YX, 
                               std::vector<variable::variable_typenames>
                               (YX.size(), variable::VECTOR_VARIABLE));
  for (size_t i(0); i < ntrain; ++i) {
    vector_assignment fa(truth.sample(rng));
    ds.insert(assignment(fa));
  }
  vector_assignment_dataset<>
    test_ds(finite_var_vector(), YX, 
            std::vector<variable::variable_typenames>
            (YX.size(), variable::VECTOR_VARIABLE));
  for (size_t i(0); i < ntest; ++i) {
    vector_assignment fa(truth.sample(rng));
    test_ds.insert(assignment(fa));
  }

  // Learn P(Y|X) via gaussian_crf_factor::learn_crf_factor.
  //  (Linear regression + empirical covariance matrix.)
  gaussian_crf_factor::parameters gcf_params;
  gcf_params.reg.lambdas[0] = .1;
  gcf_params.reg.lambdas[1] = .1;
  std::vector<gaussian_crf_factor::regularization_type> reg_params;
  vec means, stderrs;
  crossval_parameters
    cv_params(gaussian_crf_factor::regularization_type::nlambdas);
  cv_params.nfolds = 5;
  cv_params.minvals = .001;
  cv_params.maxvals = 20.;
  cv_params.nvals = 7;
  cv_params.zoom = 0;
  cv_params.log_scale = true;
  gaussian_crf_factor* f1 =
    learn_crf_factor_cv<gaussian_crf_factor>
    (reg_params, means, stderrs, cv_params, ds,
     make_domain<vector_variable>(Y),
     copy_ptr<vector_domain>(new vector_domain(X.begin(), X.end())),
     gcf_params, unif_int(rng));
  cout << "Learning CRFs using Y = " << Y << "\n"
       << " and X = " << X << endl;

  // Learn via crf_parameter_learner.
  crf_model<gaussian_crf_factor> tmp_true_model;
  tmp_true_model.add_factor(*f1);
  crf_parameter_learner<gaussian_crf_factor>::parameters cpl_params;
  cpl_params.init_iterations = 100;
  cpl_params.gm_params.convergence_zero = .00001;
  cpl_params.opt_method = real_optimizer_builder::CONJUGATE_GRADIENT;
  cpl_params.debug = 0;
  std::vector<gaussian_crf_factor::regularization_type> cpl_reg_params;
  vec cpl_means, cpl_stderrs;
  cpl_params.lambdas =
    crf_parameter_learner<gaussian_crf_factor>::choose_lambda
    (cpl_reg_params, cpl_means, cpl_stderrs, cv_params,
     tmp_true_model, false, ds, cpl_params, 0, unif_int(rng));
  crf_parameter_learner<gaussian_crf_factor>
    cpl(tmp_true_model, ds, false, cpl_params);

  // Compare the results.
  double joint_ll(0);
  foreach(const record<>& r, ds.records()) {
    joint_ll += truth.logv(r);
  }
  joint_ll /= ds.size();
  double true_ll(0);
  foreach(const record<>& r, ds.records()) {
    moment_gaussian mg(true_conditional.restrict
                       (r.assignment(make_domain<vector_variable>(X))));
    mg.normalize();
    true_ll += mg.logv(r);
  }
  true_ll /= ds.size();
  double gcf_ll(0);
  foreach(const record<>& r, ds.records()) {
    canonical_gaussian cg(f1->condition(r));
    cg.normalize();
    gcf_ll += cg.logv(r);
  }
  gcf_ll /= ds.size();
  double cpl_ll(0);
  foreach(const record<>& r, ds.records()) {
    cpl_ll += cpl.current_model().log_likelihood(r);
  }
  cpl_ll /= ds.size();

  double joint_test_ll(0);
  foreach(const record<>& r, test_ds.records()) {
    joint_test_ll += truth.logv(r);
  }
  joint_test_ll /= test_ds.size();
  double true_test_ll(0);
  foreach(const record<>& r, test_ds.records()) {
    moment_gaussian mg(true_conditional.restrict
                       (r.assignment(make_domain<vector_variable>(X))));
    mg.normalize();
    true_test_ll += mg.logv(r);
  }
  true_test_ll /= test_ds.size();
  double gcf_test_ll(0);
  foreach(const record<>& r, test_ds.records()) {
    canonical_gaussian cg(f1->condition(r));
    cg.normalize();
    gcf_test_ll += cg.logv(r);
  }
  gcf_test_ll /= test_ds.size();
  double cpl_test_ll(0);
  foreach(const record<>& r, test_ds.records()) {
    cpl_test_ll += cpl.current_model().log_likelihood(r);
  }
  cpl_test_ll /= test_ds.size();

  cout << "True conditional:\n" << true_conditional << endl;
  cout << endl;
  cout << "Learned via gaussian_crf_factor::learn_crf_factor:\n"
       << (*f1) << endl;
  cout << "... and as a moment Gaussian:\n"
       << f1->get_gaussian<moment_gaussian>() << "\n" << endl;
  cout << "Learned via CRF parameter learner:\n" << cpl.current_model() << endl;
  cout << "... and as a moment Gaussian:\n"
       << cpl.current_model().factors().front().get_gaussian<moment_gaussian>()
       << "\n" << endl;

  cout << "Joint Gaussian factor's data log likelihood: " << joint_ll << "\n"
       << "True Gaussian factor's data log likelihood: " << true_ll << "\n"
       << "Gaussian CRF factor's data log likelihood: " << gcf_ll << "\n"
       << "CRF parameter learner's data log likelihood: " << cpl_ll << "\n"
       << endl;
  cout << "Joint Gaussian factor's test log likelihood: " << joint_test_ll
       << "\n"
       << "True Gaussian factor's test log likelihood: " << true_test_ll << "\n"
       << "Gaussian CRF factor's test log likelihood: " << gcf_test_ll << "\n"
       << "CRF parameter learner's test log likelihood: " << cpl_test_ll << "\n"
       << endl;

  delete(f1);
  f1 = NULL;

  /*
    TEST:
     - Convert the true moment_gaussian P(Y|X) to a gaussian_crf_factor.
     - Compute the empirical expectation of P(Y|X) using
       gaussian_crf_factor::condition().
   */
  gaussian_crf_factor tmp_gcf(true_conditional);
  double tmp_gcf_ll(0.);
  foreach(const record<>& r, ds.records()) {
    canonical_gaussian tmpcg(tmp_gcf.condition(r));
    tmp_gcf_ll += tmpcg.normalize().logv(r);
  }
  tmp_gcf_ll /= ds.size();
  cout << "Log likelihood using a gaussian_crf_factor constructed from the "
       << "true moment_gaussian for P(Y|X): " << tmp_gcf_ll << endl;
  cout << "The forementioned gaussian_crf_factor:\n"
       << tmp_gcf << endl;

  return 0;
}
