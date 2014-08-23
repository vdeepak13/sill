#include <iostream>
#include <sstream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/crf/crf_parameter_learner.hpp>
#include <sill/learning/dataset_old/data_conversions.hpp>
#include <sill/learning/dataset_old/vector_assignment_dataset.hpp>
#include <sill/learning/crf/learn_crf_factor.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file gaussian_crf_factor_timing.cpp  Time learning conditional Gaussians.
 */

static int usage() {
  std::cout << "usage: ./gaussian_crf_factor_timing"
            << " ([ntrain] [ntest] [Ysize] [Xsize])"
            << " ([crf_parameter_learner method]) ([line search type (0/1/2)])"
            << " (random seed)"
            << std::endl;
  return 1;
}

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Dataset parameters
  size_t ntrain = 10;
  size_t ntest = 1000;
  size_t Ysize = 2;
  size_t Xsize = 2;
  unsigned oracle_seed = time(NULL);
  double b_max = 5;

  size_t cpl_method = 1;
  size_t line_search_type = 0;

  std::istringstream is;
  switch(argc) {
  case 8:
    is.clear();
    is.str(argv[7]);
    if (!(is >> oracle_seed))
      return usage();
  case 7:
    is.clear();
    is.str(argv[6]);
    if (!(is >> line_search_type))
      return usage();
  case 6:
    is.clear();
    is.str(argv[5]);
    if (!(is >> cpl_method))
      return usage();
  case 5:
    is.clear();
    is.str(argv[1]);
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
    break;
  case 1:
    break;
  default:
    return usage();
  }

  bool normalize_data = false;

  // Gaussian CRF factor cross validation parameters
  bool do_cv = true;
  crossval_params<gaussian_crf_factor::regularization_type::nlambdas> cv_params;
  cv_params.nfolds = 2;
  cv_params.minvals.fill(.001);
  cv_params.maxvals.fill(20.);
  cv_params.nvals.fill(2);
  cv_params.zoom = 1;
  cv_params.log_scale = true;

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
  moment_gaussian_generator gen(-b_max, b_max, 2.0, 0.5);
  moment_gaussian truth_YX = gen(make_domain(YX), rng);
  truth_YX.normalize();
  if (1) {
    canonical_gaussian cg1(truth_YX);
    canonical_gaussian cg2(truth_YX);
    cg2.enforce_psd(truth_YX.mean());
    if (cg1.inf_matrix() != cg2.inf_matrix())
      assert(false);
    if (cg1.inf_vector() != cg2.inf_vector())
      assert(false);
    if (cg1.log_multiplier() != cg2.log_multiplier())
      assert(false);
  }
  moment_gaussian
    truth_Y_given_X(truth_YX.conditional(make_domain<vector_variable>(X)));
  moment_gaussian truth_X(truth_YX.marginal(make_domain(X)));

  // Generate a dataset
  cout << "Sampling " << (ntrain+ntest) << " samples from the model" << endl;
  boost::shared_ptr<vector_assignment_dataset>
    ds_ptr(new vector_assignment_dataset
           (finite_var_vector(), YX, 
            std::vector<variable::variable_typenames>
            (YX.size(), variable::VECTOR_VARIABLE)));
  for (size_t i(0); i < ntrain; ++i) {
    vector_assignment fa(truth_YX.sample(rng));
    ds_ptr->insert(assignment(fa));
  }
  vector_assignment_dataset
    test_ds(finite_var_vector(), YX, 
            std::vector<variable::variable_typenames>
            (YX.size(), variable::VECTOR_VARIABLE));
  for (size_t i(0); i < ntest; ++i) {
    vector_assignment fa(truth_YX.sample(rng));
    test_ds.insert(assignment(fa));
  }
  vector_assignment_dataset orig_ds(ds_ptr->datasource_info());
  foreach(const record& r, ds_ptr->records())
    orig_ds.insert(r);
  vector_assignment_dataset orig_test_ds(test_ds.datasource_info());
  foreach(const record& r, test_ds.records())
    orig_test_ds.insert(r);

  if (normalize_data) {
    std::pair<vec, vec> means_stddevs(ds_ptr->normalize());
    test_ds.normalize(means_stddevs.first, means_stddevs.second);
  }

  cout << "Learning CRFs using Y = " << Y << "\n"
       << " and X = " << X << endl;

  // Learn P(Y|X) via gaussian_crf_factor::learn_crf_factor.
  //  (Linear regression + empirical covariance matrix.)
  gaussian_crf_factor::parameters gcf_params;
  gcf_params.reg.lambdas[0] = .01;
  gcf_params.reg.lambdas[1] = .01;
  boost::timer timer;
  std::vector<gaussian_crf_factor::regularization_type> reg_params;
  vec means, stderrs;
  gaussian_crf_factor f1;
  if (do_cv) {
    f1 = learn_crf_factor<gaussian_crf_factor>::train_cv
      (reg_params, means, stderrs, cv_params, ds_ptr,
       make_domain<vector_variable>(Y),
       copy_ptr<vector_domain>(new vector_domain(X.begin(), X.end())),
       gcf_params, unif_int(rng));
  } else {
    f1 = learn_crf_factor<gaussian_crf_factor>::train
      (ds_ptr, make_domain<vector_variable>(Y),
       copy_ptr<vector_domain>(new vector_domain(X.begin(), X.end())),
       gcf_params, unif_int(rng));
  }
  double elapsed = timer.elapsed();

  cout << "Done with learning via learn_crf_factor<gaussian_crf_factor>::train(_cv)."
       << endl;

  // Learn via crf_parameter_learner.
  crf_model<gaussian_crf_factor> tmp_true_model;
  tmp_true_model.add_factor(f1);
  crf_parameter_learner<gaussian_crf_factor>::parameters cpl_params;
  cpl_params.init_iterations = 100;
  cpl_params.convergence_zero = .00001;
  cpl_params.method = cpl_method;
  cpl_params.line_search_type = line_search_type;
  cpl_params.debug = 0;
  std::vector<gaussian_crf_factor::regularization_type> cpl_reg_params;
  vec cpl_means, cpl_stderrs;
  timer.restart();
  cpl_params.regularization =
    crf_parameter_learner<gaussian_crf_factor>::choose_lambda
    (cpl_reg_params, cpl_means, cpl_stderrs, cv_params,
     tmp_true_model, false, *ds_ptr, cpl_params, 0, unif_int(rng));
  crf_parameter_learner<gaussian_crf_factor>
    cpl(tmp_true_model, ds_ptr, false, cpl_params);
  double cpl_time(timer.elapsed());

  cout << "Done with learning via crf_parameter_learner;\n"
       << " made " << cpl.iteration() << " calls to gradient, with "
       << cpl.objective_calls_per_iteration()
       << " avg calls to objective per gradient call."
       << endl;

  // Compare the results.
  double joint_ll(0);
  foreach(const record& r, orig_ds.records()) {
    joint_ll += truth_YX.logv(r);
  }
  joint_ll /= ds_ptr->size();
  double true_ll(0);
  foreach(const record& r, orig_ds.records()) {
    moment_gaussian mg(truth_Y_given_X.restrict
                       (r.vector_assignment(make_domain<vector_variable>(X))));
    mg.normalize();
    true_ll += mg.logv(r);
  }
  true_ll /= ds_ptr->size();
  double gcf_ll(0);
  foreach(const record& r, ds_ptr->records()) {
    canonical_gaussian cg(f1.condition(r));
    cg.normalize();
    moment_gaussian mg(cg);
    if (!cg.enforce_psd(mg.mean()))
      assert(false); // REMOVE THIS LATER
    gcf_ll += cg.logv(r);
  }
  gcf_ll /= ds_ptr->size();
  double cpl_ll(0);
  foreach(const record& r, ds_ptr->records()) {
    cpl_ll += cpl.model().log_likelihood(r);
  }
  cpl_ll /= ds_ptr->size();

  double joint_test_ll(0);
  foreach(const record& r, orig_test_ds.records()) {
    joint_test_ll += truth_YX.logv(r);
  }
  joint_test_ll /= test_ds.size();
  double true_test_ll(0);
  foreach(const record& r, orig_test_ds.records()) {
    moment_gaussian mg(truth_Y_given_X.restrict
                       (r.vector_assignment(make_domain<vector_variable>(X))));
    mg.normalize();
    true_test_ll += mg.logv(r);
  }
  true_test_ll /= test_ds.size();
  double gcf_test_ll(0);
  foreach(const record& r, test_ds.records()) {
    canonical_gaussian cg(f1.condition(r));
    cg.normalize();
    gcf_test_ll += cg.logv(r);
  }
  gcf_test_ll /= test_ds.size();
  double cpl_test_ll(0);
  foreach(const record& r, test_ds.records()) {
    cpl_test_ll += cpl.model().log_likelihood(r);
  }
  cpl_test_ll /= test_ds.size();

  cout << "True P(Y|X):\n" << truth_Y_given_X << endl;
  cout << endl;
  cout << "Learned via gaussian_crf_factor::learn_crf_factor:\n"
       << f1 << endl;
  cout << "... and as a moment Gaussian:\n" << f1.get_gaussian()
       << "\n" << endl;
  cout << "Learned via CRF parameter learner:\n" << cpl.model() << endl;
  cout << "... and as a moment Gaussian:\n"
       << cpl.model().factors().front().get_gaussian()
       << "\n" << endl;

  if (do_cv) {
    cout << "CV results for Gaussian CRF factor learning:\n"
         << "lambdas: ";
    for (size_t j(0); j < reg_params.size(); ++j)
      cout << "[" << reg_params[j].lambdas[0] << ","
           << reg_params[j].lambdas[1] << "]" << " ";
    cout << "\n"
         << "means:   " << means << "\n"
         << "stderrs: " << stderrs << "\n"
         << endl;

    cout << "CV results for CRF parameter learner:\n"
         << "lambdas: ";
    for (size_t j(0); j < cpl_reg_params.size(); ++j)
      cout << "[" << cpl_reg_params[j].lambdas[0] << ","
           << cpl_reg_params[j].lambdas[1] << "]" << " ";
    cout << "\n"
         << "means:   " << cpl_means << "\n"
         << "stderrs: " << cpl_stderrs << "\n" << endl;

    size_t max_i = max_index(means);
    cout << "Gaussian CRF chose lambda = " << reg_params[max_i].lambdas
         << ", with score = " << means[max_i] << "\n";
    max_i = max_index(cpl_means);
    cout << "CRF parameter learner chose lambda = "
         << cpl_reg_params[max_i].lambdas
         << ", with score = " << cpl_means[max_i] << "\n";
  }

  cout << "\nTime for CV and training Gaussian CRF factor: " << elapsed
       << " seconds" << endl;
  cout << "Time for CV and training via CRF parameter learner: " << cpl_time
       << " seconds\n" << endl;

  cout << "CRF parameter learner made " << cpl.iteration()
       << " gradient calls and " << cpl.objective_calls_per_iteration()
       << " avg obj calls per gradient call.\n"
       << endl;

  cout << "Joint Gaussian factor P(Y,X)'s data log likelihood: " << joint_ll
       << "\n"
       << "True Gaussian factor P(Y|X)'s data log likelihood: " << true_ll
       << "\n"
       << "Gaussian CRF factor's data log likelihood: " << gcf_ll << "\n"
       << "CRF parameter learner's data log likelihood: " << cpl_ll << "\n"
       << endl;
  cout << "Joint Gaussian factor's test log likelihood: " << joint_test_ll
       << "\n"
       << "True Gaussian factor's test log likelihood: " << true_test_ll << "\n"
       << "Gaussian CRF factor's test log likelihood: " << gcf_test_ll << "\n"
       << "CRF parameter learner's test log likelihood: " << cpl_test_ll << "\n"
       << endl;

  return 0;
}
