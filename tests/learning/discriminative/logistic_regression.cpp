#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/statistics.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/syn_oracle_majority.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/logistic_regression.hpp>
#include <sill/learning/learn_factor.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Create a dataset to work with
  universe u;
  size_t niterations = 1000;
  double lambda = 1;
  /*
  size_t ntrain = 30162;
  size_t ntest = 15060;
  boost::shared_ptr<vector_dataset> ds_train_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>
    ("/Users/jbradley/data/uci/adult/adult-train.sum", u, ntrain);
  vector_dataset& ds_train = *ds_train_ptr;
  boost::shared_ptr<vector_dataset> ds_test_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>
    ("/Users/jbradley/data/uci/adult/adult-test.sum",
     ds_train.finite_list(),
     ds_train.vector_list(), ds_train.variable_type_order(), ntest);
  vector_dataset& ds_test = *ds_test_ptr;
  finite_variable* class_var = ds_train.finite_class_variables().front();
  vec means;
  vec std_devs;
  boost::tie(means,std_devs) = ds_train.normalize();
  ds_test.normalize(means,std_devs);
  */

  size_t ntrain = 5000;
  size_t ntest = 5000;
  syn_oracle_knorm::parameters oracle_params;
  oracle_params.radius = 1;
  oracle_params.std_dev = 2.5;
  oracle_params.random_seed = 489167211;
  syn_oracle_knorm knorm(create_syn_oracle_knorm(2,20,u,oracle_params));
  finite_variable* class_var = knorm.finite_class_variables().front();
  cout << knorm;
  boost::shared_ptr<vector_dataset> ds_train_ptr
    = oracle2dataset<vector_dataset>(knorm, ntrain);
  vector_dataset& ds_train = *ds_train_ptr;
  boost::shared_ptr<vector_dataset> ds_test_ptr
    = oracle2dataset<vector_dataset>(knorm, ntest);
  vector_dataset& ds_test = *ds_test_ptr;

  /*
  size_t ntrain = 500;
  size_t ntest = 500;
  syn_oracle_majority majority(create_syn_oracle_majority(9,u));
  finite_variable* class_var = majority.finite_class_variables().front();
  cout << majority;
  boost::shared_ptr<vector_dataset> ds_train_ptr
    = oracle2dataset<vector_dataset>(majority, ntrain);
  vector_dataset& ds_train = *ds_train_ptr;
  boost::shared_ptr<vector_dataset> ds_test_ptr
    = oracle2dataset<vector_dataset>(majority, ntest);
  vector_dataset& ds_test = *ds_test_ptr;
  */

  statistics stats(ds_train);

  cout << "Marginal over training set class variables:\n"
       << learn_marginal<table_factor>(make_domain(class_var), ds_train)
       << endl;

  logistic_regression_parameters lr_params;
  lr_params.lambda = lambda;
  lr_params.init_iterations = niterations;
  logistic_regression lr(stats, lr_params);

  cout << "Trained logistic regression using batch gradient descent on "
       << ntrain << " examples to get:\n"
       << "  " << lr << "\n"
       << "now testing on " << ntest << " examples" << endl;

  cout << "Test accuracy = " << lr.test_accuracy(ds_test) << endl;

  cout << "Saving logistic_regression...";
  lr.save("logistic_regression_test.txt");
  cout << "loading logistic_regression...";
  lr.load("logistic_regression_test.txt", ds_test);
  cout << "testing logistic_regression again...";

  cout << "Saved and loaded logistic regression model:\n"
       << "  " << lr << "\n"
       << "now testing on " << ntest << " examples" << endl;

  cout << "Test accuracy = " << lr.test_accuracy(ds_test) << endl;

}
