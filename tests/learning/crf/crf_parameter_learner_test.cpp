#include <iostream>

#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/log_reg_crf_factor.hpp>
#include <sill/factor/table_crf_factor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/crf/crf_parameter_learner_builder.hpp>
#include <sill/learning/crossval_builder.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/generate_datasets.hpp>
#include <sill/learning/dataset/vector_assignment_dataset.hpp>
#include <sill/model/model_products.hpp>
#include <sill/model/random.hpp>

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
         bool do_cv, const sill::vec& fixed_lambda, boost::mt11213b& rng,
         typename sill::crf_parameter_learner<F>::parameters& cpl_params,
         sill::crossval_builder& cv_builder) {

  using namespace sill;
  using namespace std;

  typedef typename F::regularization_type regularization_type;

  boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());

  // Generate a dataset
  cout << "Sampling " << (ntrain+ntest) << " training samples from the model"
       << endl;
  boost::shared_ptr<vector_assignment_dataset>
    train_ds_ptr(new vector_assignment_dataset(ds_info, ntrain));
  generate_dataset(*train_ds_ptr, YXmodel, ntrain, rng);
  vector_assignment_dataset test_ds(ds_info, ntest);
  generate_dataset(test_ds, YXmodel, ntest, rng);

  double true_train_ll = YgivenXmodel.expected_log_likelihood(*train_ds_ptr);
  double true_test_ll = YgivenXmodel.expected_log_likelihood(test_ds);

  cout << "Doing parameter learning" << endl;
  vec means, stderrs;
  std::vector<regularization_type> reg_params;
  if (do_cv) {
    crossval_parameters<regularization_type::nlambdas>
      cv_params(cv_builder.get_parameters<regularization_type::nlambdas>());
    cv_params.nfolds = std::min(ntrain, cv_params.nfolds);
    cpl_params.lambdas =
      crf_parameter_learner<F>::choose_lambda
      (reg_params, means, stderrs, cv_params, YgivenXmodel, false,
       *train_ds_ptr, cpl_params, 0, unif_int(rng));
    cout << "Used cross-validation to choose lambdas = "
         << cpl_params.lambdas << endl;
  } else {
    cpl_params.lambdas = fixed_lambda;
  }
  crf_parameter_learner<F>
    param_learner(YgivenXmodel, train_ds_ptr, false, cpl_params);
  double train_ll = 0.;
  foreach(const assignment& a, train_ds_ptr->assignments())
    train_ll += param_learner.current_model().log_likelihood(a);
  train_ll /= train_ds_ptr->size();
  double test_ll = 0.;
  foreach(const assignment& a, test_ds.assignments())
    test_ll += param_learner.current_model().log_likelihood(a);
  test_ll /= test_ds.size();

  if (do_cv) {
    cout << "Cross-validation results:\n"
         << "lambdas: ";
    foreach(const regularization_type& reg, reg_params)
      cout << reg.lambdas << " ";
    cout << "\n"
         << "means: " << means << "\n"
         << "stderrs: " << stderrs << "\n"
         << "Chose lambdas = " << cpl_params.lambdas << "\n"
         << endl;
  }

  cout << "crf_parameter_learner made " << param_learner.iteration()
       << " calls to gradient, with "
       << param_learner.objective_calls_per_iteration()
       << " avg calls to objective per gradient call."
       << endl;

  cout << "True model's avg training data log likelihood: " << true_train_ll
       << endl;
  cout << "True model's avg test data log likelihood: " << true_test_ll << endl;

  cout << "CRF's avg training data log likelihood after parameter learning: "
       << train_ll << endl;
  cout << "CRF's avg test data log likelihood after parameter learning: "
       << test_ll << endl;
}

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Parse the command-line parameters
  //  Dataset parameters
  size_t ntrain;
  size_t ntest;
  size_t model_size;
  std::string model_type;
  std::string factor_type;
  //  CRF parameter learner parameters
  crf_parameter_learner_builder cpl_builder;
  //  CV parameters
  bool do_cv;
  vec fixed_lambda; // if not doing CV
  crossval_builder cv_builder;
  //  Other parameters
  unsigned random_seed;

  namespace po = boost::program_options;
  po::options_description
    desc("Allowed options for crf_parameter_learner_test");

  desc.add_options()
    ("help", "Print help message.");
  //  Dataset parameters
  desc.add_options()
    ("ntrain", po::value<size_t>(&ntrain)->default_value(10),
     "Number of training examples")
    ("ntest", po::value<size_t>(&ntest)->default_value(100),
     "Number of test examples")
    ("model_size", po::value<size_t>(&model_size)->default_value(10),
     "Number of Y (and X) variables")
    ("model_type", po::value<std::string>(&model_type)->default_value("chain"),
     "Model type: chain, tree.")
    ("factor_type",
     po::value<std::string>(&factor_type)->default_value("table"),
     "Factor type: table, gaussian.");
  //  CV parameters
  desc.add_options()
    ("do_cv", po::bool_switch(&do_cv),
     "Do cross validation to choose lambdas for parameter learning.")
    ("fixed_lambda", po::value<vec>(&fixed_lambda)->default_value(vec(1,0)),
     "Fixed lambda (1 or 2 values, depending on factor type)");
  //  Other parameters
  desc.add_options()
    ("random_seed",
     po::value<unsigned>(&random_seed)->default_value(time(NULL)),
     "Random seed. (default=time)");
  //  CRF parameter learner parameters
  cpl_builder.add_options(desc);
  cv_builder.add_options(desc);

  // Parse options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Check options
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  //  Dataset parameters
  if (ntrain == 0 ||
      ntest == 0 ||
      model_size == 0 ||
      !(model_type == "chain" || model_type == "tree") ||
      !(factor_type == "table" || factor_type == "gaussian")) {
    cout << desc << endl;
    return 1;
  }
  //  CRF parameter learner parameters
  crf_parameter_learner_parameters cpl_params(cpl_builder.get_parameters());
  if (!cpl_params.valid()) {
    cout << desc << endl;
    return 1;
  }

  boost::mt11213b rng(random_seed);
  boost::uniform_int<int> unif_int(0,std::numeric_limits<int>::max());

  cpl_params.random_seed = unif_int(rng);

  // Create a model
  universe u;
  if (factor_type == "table") {
    decomposable<table_factor> YXmodel;
    crf_model<table_crf_factor> YgivenXmodel;
    boost::tuple<finite_var_vector, finite_var_vector,
      std::map<finite_variable*, copy_ptr<finite_domain> > >
      Y_X_and_map(create_random_crf(YXmodel, YgivenXmodel, model_size, 2, u,
                                    model_type, "associative", 2, 2, 2,
                                    false, unif_int(rng)));
    model_product_inplace(YgivenXmodel, YXmodel);
    finite_var_vector Y(Y_X_and_map.get<0>());
    finite_var_vector X(Y_X_and_map.get<1>());
    finite_var_vector YX(Y);
    YX.insert(YX.end(), X.begin(), X.end());
    cout << "True model for P(Y,X):\n" << YXmodel << "\n" << endl;
    datasource_info_type ds_info;
    ds_info.finite_seq = YX;
    ds_info.var_type_order =
      std::vector<variable::variable_typenames>
      (YX.size(), variable::FINITE_VARIABLE);

    run_test<table_crf_factor>
      (YgivenXmodel, YXmodel, ds_info, ntrain, ntest, do_cv, fixed_lambda, rng,
       cpl_params, cv_builder);
  } else if (factor_type == "gaussian") {
    decomposable<canonical_gaussian> YXmodel;
    crf_model<gaussian_crf_factor> YgivenXmodel;
    boost::tuple<vector_var_vector, vector_var_vector,
      std::map<vector_variable*, copy_ptr<vector_domain> > >
      Y_X_and_map(create_random_gaussian_crf
                  (YXmodel, YgivenXmodel, model_size, u,
                   model_type, 2, 2, 1, .3, .25, .1,
                   false, unif_int(rng)));
    model_product_inplace(YgivenXmodel, YXmodel);
    vector_var_vector Y(Y_X_and_map.get<0>());
    vector_var_vector X(Y_X_and_map.get<1>());
    vector_var_vector YX(Y);
    YX.insert(YX.end(), X.begin(), X.end());
    cout << "True model for P(Y,X):\n" << YXmodel << "\n" << endl;
    datasource_info_type ds_info;
    ds_info.vector_seq = YX;
    ds_info.var_type_order =
      std::vector<variable::variable_typenames>
      (YX.size(), variable::VECTOR_VARIABLE);

    run_test<gaussian_crf_factor>
      (YgivenXmodel, YXmodel, ds_info, ntrain, ntest, do_cv, fixed_lambda, rng,
       cpl_params, cv_builder);
  } else {
    assert(false);
  }

  return 0;
}
