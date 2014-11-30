#include <sill/factor/table_factor.hpp>
#include <sill/model/naive_bayes.hpp>
#include <sill/learning/dataset/finite_memory_dataset.hpp>
#include <sill/learning/dataset/finite_dataset_io.hpp>
#include <sill/learning/parameter/naive_bayes_mle.hpp>
#include <sill/learning/validation/cross_validation.hpp>

int main(int argc, char** argv) {
  using namespace sill;
  std::string data_dir = argc > 1 ? argv[1] : "";
  size_t num_folds = 10;

  // load the dataset format
  universe u;
  symbolic_format format;
  format.load_config(data_dir + "car.cfg", u);

  // load the dataset
  finite_memory_dataset ds;
  load(data_dir + "car.data", format, ds);

  // create the naive Bayes learner with the label variable named "class"
  // and the remaining variables in the dataset being features
  finite_variable* label = format.finite_var("class");
  finite_var_vector features = format.finite_var_vec();
  features.erase(std::find(features.begin(), features.end(), label));
  naive_bayes_mle<table_factor> learner(label, features);

  // train the model on all of the data,
  // using the default regularization parameters (set to 0)
  naive_bayes<table_factor> model;
  double ll_model = learner.learn(ds, model);
  double ll_prior = model.prior().log_likelihood(ds);
  double ll_cond  = model.conditional_log_likelihood(ds);
  std::cout << "Log-likelihood of the model: " << ll_model << std::endl
            << "Log-likelihood of the prior: " << ll_prior << std::endl
            << "Conditional log-likelihood:  " << ll_cond << std::endl;

  // compute the views for the k-fold cross-validation
  boost::mt19937 rng;
  ds.shuffle(rng); // order the rows randomly
  double ll_cond2 = model.conditional_log_likelihood(ds);
  std::cout << "Verify cond. log-likelihood: " << ll_cond2 << std::endl;
  std::vector<slice_view<finite_dataset> > train, test;
  kfold_split(ds, num_folds, train, test); // for each fold, extract train&test

  // compute the k-fold cross-validation error
  double ll_test = 0.0;
  for (size_t k = 0; k < num_folds; ++k) {
    learner.learn(train[k], model);
    ll_test += model.conditional_log_likelihood(test[k]);
  }
  std::cout << num_folds << "-fold CV log-likelihood: " << ll_test << std::endl;

  return 0;
}
