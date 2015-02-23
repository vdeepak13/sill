#include <sill/base/universe.hpp>
#include <sill/learning/dataset/hybrid_dataset_io.hpp>
#include <sill/learning/dataset/hybrid_memory_dataset.hpp>
#include <sill/learning/dataset/symbolic_format.hpp>
#include <sill/learning/parameter/logistic_regression.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char** argv) {
  using namespace sill;
  std::string data_dir = argc > 1 ? argv[1] : "";

  // load the dataset format
  universe u;
  symbolic_format format;
  format.load_config(data_dir + "iris.cfg", u);

  // load the dataset
  hybrid_memory_dataset<> ds;
  load(data_dir + "iris.data", format, ds);
  std::cout << ds << std::endl;

  // create the logistic regression learner with the label variable "class"
  // and the vector variables in the dataset being the features
  finite_variable* label = format.finite_var("class");
  vector_var_vector features = format.vector_var_vec();
  logistic_regression<>::param_type params;
  params.max_iter = argc > 2 ? atoi(argv[2]) : 10;
  params.regul = argc > 3 ? atof(argv[3]) : 0.1;
  logistic_regression<> learner(label, features);

  // train the model and output some statistics
  softmax<> model;
  double ll = learner.learn(ds, params, model);
  std::cout << model << std::endl;
  std::cout << "log-likelihood: " << ll << std::endl;
  std::cout << "accuracy: " << model.accuracy(ds)<< std::endl;
}
