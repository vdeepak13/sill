#include <iostream>

#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/argument/universe.hpp>
#include <sill/learning/crf/crf_parameter_learner_builder.hpp>
#include <sill/learning/crf/crf_validation_functor.hpp>
#include <sill/learning/crf/pwl_crf_parameter_learner.hpp>
#include <sill/learning/validation/crossval_builder.hpp>
#include <sill/learning/validation/validation_framework.hpp>
#include <sill/learning/dataset_old/data_conversions.hpp>
#include <sill/learning/dataset_old/generate_datasets.hpp>
#include <sill/learning/dataset_old/vector_dataset.hpp>
#include <sill/model/model_products.hpp>
#include <sill/model/random_crf_builder.hpp>

#include <sill/macros_def.hpp>

/**
 * Run the CRF parameter learner test.
 */
template <typename F>
void
run_test(const sill::crf_model<F>& YgivenXmodel,
         const sill::decomposable<typename F::output_factor_type>& YXmodel,
         const sill::datasource_info_type& ds_info,
         size_t ntrain, size_t ntest,
         const sill::vec& fixed_lambda, boost::mt11213b& rng,
         typename sill::crf_parameter_learner<F>::parameters& cpl_params,
         sill::crossval_builder& cv_builder, bool init_with_pwl) {

  using namespace sill;
  using namespace std;

  typedef typename F::regularization_type regularization_type;

  boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());

  cout << "True model for P(Y,X):\n" << YXmodel << "\n" << endl;

  // Generate a dataset
  cout << "Sampling " << ntrain << " training samples and "
       << ntest << " test samples from the model" << endl;
  vector_dataset_old<> train_ds(ds_info, ntrain);
  generate_dataset(train_ds, YXmodel, ntrain, rng);
  vector_dataset_old<> test_ds(ds_info, ntest);
  generate_dataset(test_ds, YXmodel, ntest, rng);

  double true_train_ll = train_ds.expected_value(YgivenXmodel.log_likelihood());
  double true_test_ll = test_ds.expected_value(YgivenXmodel.log_likelihood());

  cout << "Doing parameter learning" << endl;
  crf_model<F> init_model;
  if (init_with_pwl) {
    typename pwl_crf_parameter_learner<F>::parameters pcpl_params;
    pcpl_params.crf_factor_cv = false;
//    pcpl_params.cv_params.nfolds = 2;
//    pcpl_params.cv_params.nvals = 4;
    pcpl_params.random_seed = unif_int(rng);
    pwl_crf_parameter_learner<F> pcpl(train_ds, YgivenXmodel, pcpl_params);
    init_model = pcpl.model();
  }
  if (cv_builder.no_cv) {
    cpl_params.lambdas = fixed_lambda;
  } else {
    crossval_parameters
      cv_params(cv_builder.get_parameters(regularization_type::nlambdas));
    size_t score_type = 0; // log likelihood
    cpl_params.lambdas =
      crf_parameter_learner<F>::choose_lambda
      (cv_params, (init_with_pwl ? init_model : YgivenXmodel), init_with_pwl,
       train_ds, cpl_params, score_type, unif_int(rng));

    cout << "Cross-validation chose lambda: " << cpl_params.lambdas << endl;
  }
  cpl_params.random_seed = unif_int(rng);
  crf_model<F> learned_model;
  size_t cpl_iterations;
  size_t cpl_obj_calls_per_iter;
  if (init_with_pwl) {
    crf_parameter_learner<F>
      param_learner(init_model, false, train_ds, cpl_params);
    learned_model = param_learner.model();
    cpl_iterations = param_learner.iteration();
    cpl_obj_calls_per_iter =
      param_learner.my_objective_count() / param_learner.iteration();
  } else {
    crf_parameter_learner<F>
      param_learner(YgivenXmodel, true,
                    train_ds, cpl_params);
    learned_model = param_learner.model();
    cpl_iterations = param_learner.iteration();
    cpl_obj_calls_per_iter =
      param_learner.my_objective_count() / param_learner.iteration();
  }
  double train_ll = train_ds.expected_value(learned_model.log_likelihood());
  double test_ll = test_ds.expected_value(learned_model.log_likelihood());

  cout << "crf_parameter_learner did " << cpl_iterations
       << " iterations, with " << cpl_obj_calls_per_iter
       << " avg calls to objective per iteration."
       << endl;

  cout << "True model's avg training data log likelihood: " << true_train_ll
       << endl;
  cout << "True model's avg test data log likelihood: " << true_test_ll << endl;

  cout << "CRF's avg training data log likelihood after parameter learning: "
       << train_ll << endl;
  cout << "CRF's avg test data log likelihood after parameter learning: "
       << test_ll << endl;
} // run_test


int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Parse the command-line parameters
  //  Model type
  random_crf_builder crf_builder;
  //  Dataset
  size_t ntrain;
  size_t ntest;
  //  CRF parameter learner
  crf_parameter_learner_builder cpl_builder;
  //  CV
  crossval_builder cv_builder;
  //  Other parameters
  bool init_with_pwl;
  unsigned random_seed;

  namespace po = boost::program_options;
  po::options_description
    desc("Allowed options for crf_parameter_learner_test2");

  desc.add_options()
    ("help", "Print help message.");
  //  Dataset parameters
  desc.add_options()
    ("ntrain", po::value<size_t>(&ntrain)->default_value(10),
     "Number of training examples")
    ("ntest", po::value<size_t>(&ntest)->default_value(100),
     "Number of test examples");
  //  Other parameters
  desc.add_options()
    ("init_with_pwl",
     po::bool_switch(&init_with_pwl),
     "If set, initialize parameters with piecewise likelihood estimates.")
    ("random_seed",
     po::value<unsigned>(&random_seed)->default_value(time(NULL)),
     "Random seed. (default=time)");
  //  CRF parameter learner parameters
  crf_builder.add_options(desc);
  cpl_builder.add_options(desc);
  cv_builder.add_options(desc);

  // Parse options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Check options
  if (vm.count("help") ||
      ntrain == 0 || ntest == 0) {
    cout << desc << endl;
    return 1;
  }
  //  CRF parameter learner parameters
  crf_parameter_learner_parameters cpl_params(cpl_builder.get_parameters());
  cpl_params.check();

  boost::mt11213b rng(random_seed);
  boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());

  cpl_params.random_seed = unif_int(rng);

  // Create a model
  universe u;
  if (crf_builder.factor_type == "table") {
    decomposable<table_factor> YXmodel;
    crf_model<table_crf_factor> YgivenXmodel;
    domain Y, X;
    std::map<variable, copy_ptr<finite_domain> > Y2X_map;
    crf_builder.create_model(YXmodel, YgivenXmodel, Y, X, Y2X_map,
                             u, unif_int(rng));
    model_product_inplace(YgivenXmodel, YXmodel);
    domain YX(sill::concat(Y,X));
    datasource_info_type ds_info(YX);

    run_test<table_crf_factor>
      (YgivenXmodel, YXmodel, ds_info, ntrain, ntest, cv_builder.fixed_vals,
       rng, cpl_params, cv_builder, init_with_pwl);
  } else if (crf_builder.factor_type == "gaussian") {
    decomposable<canonical_gaussian> YXmodel;
    crf_model<gaussian_crf_factor> YgivenXmodel;
    domain Y, X;
    std::map<variable, copy_ptr<vector_domain> > Y2X_map;
    crf_builder.create_model(YXmodel, YgivenXmodel, Y, X, Y2X_map,
                             u, unif_int(rng));
    model_product_inplace(YgivenXmodel, YXmodel);
    domain YX(sill::concat(Y,X));
    datasource_info_type ds_info(YX);

    run_test<gaussian_crf_factor>
      (YgivenXmodel, YXmodel, ds_info, ntrain, ntest, cv_builder.fixed_vals,
       rng, cpl_params, cv_builder, init_with_pwl);
  } else {
    assert(false);
  }

  return 0;
}
