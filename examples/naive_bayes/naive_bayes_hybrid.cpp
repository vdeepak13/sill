#include <sill/factor/hybrid.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/model/naive_bayes.hpp>
#include <sill/learning/dataset/hybrid_memory_dataset.hpp>
#include <sill/learning/dataset/hybrid_dataset_io.hpp>
#include <sill/learning/parameter/naive_bayes_mle.hpp>
#include <sill/learning/parameter/naive_bayes_em.hpp>

#include <sill/macros_def.hpp>

void head(std::ostream& out, const sill::hybrid_dataset<>& ds, size_t cnt = 10) {
  size_t i = 0;
  foreach (const sill::hybrid_record<>& r, ds.records(ds.arg_vector())) {
    if (i++ >= cnt) { break; }
    out << r << std::endl;
  }
}

void transform_log(sill::hybrid_dataset<>& ds, sill::vector_variable* var) {
  foreach (sill::vector_record<>& r, ds.records(make_vector(var))) {
    r.values = log(r.values + 1.0);
  }
}

int main(int argc, char** argv) {
  using namespace sill;
  std::string data_dir = argc > 1 ? argv[1] : "";
  typedef hybrid<moment_gaussian> hybrid_moment;

  // load the dataset format and extract the variables
  universe u;
  symbolic_format format;
  format.load_config(data_dir + "adult.cfg", u);
  finite_variable* label = format.finite_var("income");
  var_vector features = format.all_var_vec();
  features.erase(std::find(features.begin(), features.end(), label));

  // load the datasets
  hybrid_memory_dataset<> train_ds;
  hybrid_memory_dataset<> test_ds; 
  load(data_dir + "adult.data", format, train_ds);
  format.skip_rows = 1;
  load(data_dir + "adult.test", format, test_ds);

  // train a naive Bayes classifier using all the observed data
  naive_bayes_mle<hybrid_moment>::param_type mle_params;
  mle_params.feature.smoothing = 5.0;
  naive_bayes_mle<hybrid_moment> mle(label, features);
  naive_bayes<hybrid_moment> mle_model;
  double mle_ll = mle.learn(train_ds, mle_params, mle_model);
  //std::cout << "Model: " << mle_model << std::endl;
  std::cout << "MLE: " << std::endl
            << "train ll = " << mle_ll << std::endl
            << "test ll = " << mle_model.log_likelihood(test_ds) << std::endl
            << "train acc = " << mle_model.accuracy(train_ds) << std::endl
            << "test acc = " << mle_model.accuracy(test_ds) << std::endl;

  // train a naive Bayes cluster
  naive_bayes_em<hybrid_moment>::param_type em_params;
  em_params.feature_params.smoothing = 5.0;
  em_params.feature_params.comp_params = 5.0;
  em_params.verbose = true;
  if (argc > 2) { em_params.max_iters = atoi(argv[2]); }
  naive_bayes_em<hybrid_moment> em(label, features);
  naive_bayes<hybrid_moment> em_model;
  double em_bound = em.learn(train_ds, em_params, em_model);
  std::cout << "Model: " << em_model << std::endl;
  double train_acc = em_model.accuracy(train_ds);
  double test_acc = em_model.accuracy(test_ds);
  if (train_acc < 0.5) {
    train_acc = 1.0 - train_acc;
    test_acc = 1.0 - test_acc;
  }
  std::cout << "EM: " << std::endl
            << "train ll >= " << em_bound << std::endl
            << "train acc = " << train_acc << std::endl
            << "test acc = " << test_acc << std::endl;

  return 0;
}
