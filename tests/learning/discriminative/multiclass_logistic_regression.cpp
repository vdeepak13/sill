#include <iostream>

#include <boost/program_options.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/statistics.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/syn_oracle_majority.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/multiclass_logistic_regression.hpp>
#include <sill/learning/learn_factor.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  //==========================================================

  // Dataset options: synthetic data
  std::string synthetic_data;
  size_t ntrain;
  size_t ntest;
  // Dataset options: data from a file
  std::string data_path;
  double fraction_train;

  namespace po = boost::program_options;
  po::options_description
    desc(std::string("Allowed options"));

  // Dataset options
  desc.add_options()
    ("synthetic_data", po::value<std::string>(&synthetic_data),
     "Use a synthetic dataset (knorm/majority)")
    ("ntrain", po::value<size_t>(&ntrain),
     "Number of training samples (if using synthetic data)")
    ("ntest", po::value<size_t>(&ntest),
     "Number of test samples (if using synthetic data)")
    ("data_path", po::value<std::string>(&data_path),
     "Use a dataset specified by this filepath")
    ("fraction_train", po::value<double>(&fraction_train),
     "Fraction of data to use for training, with the rest used for testing (if using data from a file)");

  // Parse options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Check options
  if (!(vm.count("synthetic_data") || vm.count("data_path"))) {
    cout << desc << endl;
    return 1;
  }

  //==========================================================

  // Learning parameters
  size_t niterations = 1000;
  size_t method = 2;
  size_t regularization = 1;
  bool normalize_data = true;
  crossval_parameters<1> cv_params;
  cv_params.nfolds = 10;
  cv_params.minvals = .001;
  cv_params.maxvals = 100;
  cv_params.nvals = 10;
  cv_params.zoom = 1;
  cv_params.log_scale = true;
  unsigned cv_seed = 45898432;

  // Create a dataset to work with
  universe u;

  boost::shared_ptr<vector_dataset> ds_train_ptr;
  boost::shared_ptr<vector_dataset> ds_test_ptr;

  if (vm.count("synthetic_data")) {
    if (synthetic_data == "knorm") {
      ntrain = 5000;
      ntest = 5000;
      syn_oracle_knorm::parameters oracle_params;
      oracle_params.radius = 1;
      oracle_params.std_dev = 2.5;
      oracle_params.random_seed = 489167211;
      syn_oracle_knorm knorm(create_syn_oracle_knorm(3,20,u,oracle_params));
      cout << knorm;
      ds_train_ptr = oracle2dataset<vector_dataset>(knorm, ntrain);
      ds_test_ptr = oracle2dataset<vector_dataset>(knorm, ntest);
    } else if (synthetic_data == "majority") {
      ntrain = 1000;
      ntest = 1000;
      syn_oracle_majority::parameters oracle_params;
      oracle_params.r_vars = .3;
      oracle_params.feature_noise = .1;
      oracle_params.label_noise = .1;
      oracle_params.random_seed = 489167211;
      syn_oracle_majority
        majority(create_syn_oracle_majority(20, u, oracle_params));
      cout << majority;
      ds_train_ptr = oracle2dataset<vector_dataset>(majority, ntrain);
      ds_test_ptr = oracle2dataset<vector_dataset>(majority, ntest);
    } else {
      assert(false);
    }
  } else {
    boost::shared_ptr<vector_dataset> ds_ptr =
      data_loader::load_symbolic_dataset<vector_dataset>(data_path, u);
    ds_train_ptr.reset(new vector_dataset(ds_ptr->datasource_info()));
    ds_test_ptr.reset(new vector_dataset(ds_ptr->datasource_info()));
    ds_ptr->randomize();
    for (size_t i(0); i < (size_t)(ds_ptr->size() * fraction_train); ++i)
      ds_train_ptr->insert(ds_ptr->operator[](i));
    for (size_t i((size_t)(ds_ptr->size() * fraction_train));
         i < ds_ptr->size(); ++i)
      ds_test_ptr->insert(ds_ptr->operator[](i));
    ntrain = ds_train_ptr->size();
    ntest = ds_test_ptr->size();
    vec means;
    vec std_devs;
    if (normalize_data) {
      boost::tie(means,std_devs) = ds_train_ptr->normalize();
      ds_test_ptr->normalize(means,std_devs);
    }
  }

  vector_dataset& ds_train = *ds_train_ptr;
  vector_dataset& ds_test = *ds_test_ptr;
  finite_variable* class_var = ds_train.finite_class_variables().front();

  statistics stats(ds_train);

  cout << "Marginal over training set class variables:\n"
       << learn_marginal<table_factor>(make_domain(class_var), ds_train)
       << endl;

  multiclass_logistic_regression_parameters lr_params;
  lr_params.init_iterations = niterations;
  lr_params.regularization = regularization;
  lr_params.method = method;
  std::vector<vec> lambdas;
  vec means, stderrs;
  lr_params.lambda =
    multiclass_logistic_regression::choose_lambda
    (lambdas, means, stderrs, cv_params, ds_train_ptr, lr_params, cv_seed);
  cout << "Chose lambda via " << cv_params.nfolds << "-fold CV:\n"
       << "Lambdas:\t" << lambdas << "\n"
       << "Loglikes:\t" << means << "\n"
       << "Stderrs:\t" << stderrs << "\n"
       << "Chose lambda = " << lr_params.lambda << "\n"
       << std::endl;
  multiclass_logistic_regression lr(stats, lr_params);

  cout << "Trained multiclass logistic regression via batch gradient descent"
       << " on " << ntrain << " examples to get:\n"
       << "  " << lr << "\n"
       << "Now testing on " << ntest << " examples" << endl;

  cout << "Test accuracy = " << lr.test_accuracy(ds_test) << endl
       << "Test [log likelihood, std dev] = "
       << lr.test_log_likelihood(ds_test) << endl;

  /*
  cout << "Saving multiclass_logistic_regression...";
  lr.save("multiclass_logistic_regression_test.txt");
  cout << "loading multiclass_logistic_regression...";
  lr.load("multiclass_logistic_regression_test.txt", ds_test);
  cout << "testing multiclass_logistic_regression again...";

  cout << "Saved and loaded multiclass logistic regression:\n"
       << "  " << lr << "\n"
       << "Now testing on " << ntest << " examples" << endl;

  cout << "Test accuracy = " << lr.test_accuracy(ds_test) << endl;
  */

}
