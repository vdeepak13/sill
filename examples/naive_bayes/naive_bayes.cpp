#include <sill/factor/table_factor.hpp>
#include <sill/model/naive_bayes.hpp>
#include <sill/learning/dataset/finite_memory_dataset.hpp>
#include <sill/learning/dataset/finite_dataset_io.hpp>
#include <sill/learning/factor_mle/table_factor.hpp>
#include <sill/learning/parameter/naive_bayes_learner.hpp>

int main(int argc, char** argv) {
  using namespace sill;
  std::string data_dir = argc > 1 ? argv[1] : "";

  // load the dataset format
  universe u;
  symbolic_format format;
  format.load_config(data_dir + "car.cfg", u);

  // load the dataset
  finite_memory_dataset ds;
  load(data_dir + "car.data", format, ds);

  // create the naive Bayes learner with the label variable named "class"
  // and the remaining variables in the dataset being features
  finite_variable* label_var = format.finite_var("class");
  finite_domain feature_vars = format.finite_vars();
  feature_vars.erase(label_var);
  naive_bayes_learner<table_factor> learner(label_var, feature_vars);

  // train the model on all of the data,
  // using the default regularization parameters (set to 0)
  naive_bayes<table_factor> model;
  double log_likelihood = learner.learn(ds, model);
  std::cout << "Log-likelihood of the model: " << log_likelihood << std::endl;
  
//   // shuffle the 
//   boost::mt19937 rng;
//   ds.shuffle(rng);

// todo trim the values
  
  return 0;
}
