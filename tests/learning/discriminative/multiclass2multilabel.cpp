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

  size_t ntrain;
  size_t ntest;
  boost::shared_ptr<vector_dataset<> > ds_train_ptr;
  boost::shared_ptr<vector_dataset<> > ds_test_ptr;

  ntrain = 30162;
  ntest = 15060;
  ds_train_ptr = data_loader::load_symbolic_dataset<vector_dataset<> >
    ("/Users/jbradley/data/uci/adult/adult-train.sum", u, ntrain);
  finite_var_vector new_class_vars(ds_train_ptr->finite_class_variables());
  new_class_vars.push_back(ds_train_ptr->finite_list()[6]);
  // 0: workclass (arity 8)
  // 6: sex (arity 2)
  ds_train_ptr->set_finite_class_variables(new_class_vars);
  ds_test_ptr = data_loader::load_symbolic_dataset<vector_dataset<> >
    ("/Users/jbradley/data/uci/adult/adult-test.sum",
     ds_train_ptr->datasource_info(), ntest);
  ds_test_ptr->set_finite_class_variables(new_class_vars);
  vec means;
  vec std_devs;
  if (normalize_data) {
    boost::tie(means,std_devs) = ds_train_ptr->normalize();
    ds_test_ptr->normalize(means,std_devs);
  }
  vector_dataset<>& ds_train = *ds_train_ptr;
  vector_dataset<>& ds_test = *ds_test_ptr;

  size_t new_class_vars_size(1);
  foreach(finite_variable* v, new_class_vars)
    new_class_vars_size *= v->size();
  finite_variable* new_merged_var = u.new_finite_variable(new_class_vars_size);

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
