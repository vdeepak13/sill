#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/syn_oracle_majority.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
//#include <sill/learning/discriminative/concepts.hpp>
//#include <sill/learning/discriminative/discriminative.hpp>
#include <sill/learning/discriminative/multiclass_logistic_regression.hpp>
#include <sill/learning/discriminative/multiclass2multilabel.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  bool normalize_data = true;

  // Create a dataset to work with
  universe u;
  syn_oracle_majority majority(create_syn_oracle_majority(100,u));
  finite_variable* class_var = majority.finite_class_variables().front();

  size_t ntrain = 500;
  size_t ntest = 500;

  vector_dataset<> ds_train;
  oracle2dataset(majority, ntrain, ds_train);
  vector_dataset<> ds_test;
  oracle2dataset(majority, ntest, ds_test);

  finite_var_vector new_class_vars;
  new_class_vars.push_back(ds_train.finite_list()[0]);
  new_class_vars.push_back(ds_train.finite_list()[1]);
  ds_train.set_finite_class_variables(new_class_vars);
  ds_test.set_finite_class_variables(new_class_vars);
  vec means;
  vec std_devs;
  if (normalize_data) {
    boost::tie(means,std_devs) = ds_train.normalize();
    ds_test.normalize(means,std_devs);
  }

  finite_variable* new_merged_var =
    u.new_finite_variable(num_assignments(make_domain(new_class_vars)));

  dataset_statistics<> stats(ds_train);

  multiclass_logistic_regression_parameters mlr_params;
  mlr_params.init_iterations = ntrain * 10;
  mlr_params.regularization = 2;
  mlr_params.lambda = .5;
  mlr_params.opt_method = real_optimizer_builder::CONJUGATE_GRADIENT;
  multiclass2multilabel_parameters<> baseparams;
  baseparams.base_learner =
    boost::shared_ptr<multiclass_classifier<> >
    (new multiclass_logistic_regression<>(mlr_params));
  baseparams.random_seed = 68739024;
  baseparams.new_label = new_merged_var;
  multiclass2multilabel<> mlr(stats, baseparams);

  cout << "Trained multilabel logistic regression via batch gradient descent"
       << " on " << ntrain << " examples.\n"
       << "Now testing on " << ntest << " examples" << endl;

  cout << "Per-label test accuracy = " << mlr.test_accuracy(ds_test) << endl
       << "Test <log likelihood, std dev> = "
       << mlr.test_log_likelihood(ds_test) << endl;

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
