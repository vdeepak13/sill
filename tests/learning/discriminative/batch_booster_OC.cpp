#include <iostream>

#include <prl/base/universe.hpp>
#include <prl/learning/dataset/dataset_view.hpp>
#include <prl/learning/dataset/data_loader.hpp>
#include <prl/learning/dataset/statistics.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/discriminative/batch_booster_OC.hpp>
#include <prl/learning/discriminative/boosters.hpp>
#include <prl/learning/discriminative/concepts.hpp>
#include <prl/learning/discriminative/stump.hpp>
#include <prl/learning/learn_factor.hpp>

#include <prl/macros_def.hpp>

/**
 * Test AdaBoost OC on letter dataset from UCI ML Repository.
 * This performs about the same as the (approximately) identical test done in
 *  Schapire 1997 (ICML).
 */
int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  size_t ntrain = 5000;
  size_t ntest = 1000;
  size_t niterations = 100;
  double random_seed = 340891347;
  size_t resampling = 500;

  // Load a dataset to work with
  universe u;
  boost::shared_ptr<vector_dataset> ds_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>
    ("/Users/jbradley/data/uci/letter/letter.sum", u, ntrain + ntest);
  
  ds_ptr->randomize(29875123);
  dataset_view ds_train(*ds_ptr);
  ds_train.set_record_range(0, ntrain);
  dataset_view ds_test(*ds_ptr);
  ds_test.set_record_range(ntrain, ntrain + ntest);
  statistics stats_train(ds_train);

  finite_variable* class_var = ds_train.finite_class_variables().front();
  assert(class_var->size() == 26);
  size_t class_var_index = ds_train.record_index(class_var);
  finite_variable* binary_var = u.new_finite_variable(2);

  cout << "Marginal (from training set) over class variable:\n"
       << learn_marginal<table_factor>(make_domain(class_var), ds_train)
       << endl;

  typedef stump<> wl_type;
  boost::shared_ptr<wl_type> wl_ptr(new wl_type());
  typedef batch_booster_OC<boosting::adaboost> batch_booster_type;
  batch_booster_OC_parameters batch_booster_params;
  batch_booster_params.init_iterations = niterations;
  batch_booster_params.random_seed = random_seed;
  batch_booster_params.binary_label = binary_var;
  batch_booster_params.resampling = resampling;
  batch_booster_params.weak_learner = wl_ptr;

  if (resampling > 0)
    cout << "Running AdaBoost OC with resampling..." << endl;
  else
    cout << "Running AdaBoost OC without resampling..." << endl;
  batch_booster_type booster(stats_train, batch_booster_params);
  double accuracy = 0;
  for (size_t i = 0; i < ds_test.size(); ++i)
    if (booster.predict(ds_test[i]) == ds_test.finite(i,class_var_index))
      ++accuracy;
  accuracy /= ds_test.size();
  cout << "Test accuracy on " << ntest << " test points: " << accuracy << endl;

  cout << "Saving batch_booster_OC...";
  booster.save("batch_booster_OC_test.txt");
  cout << "loading batch_booster_OC...";
  booster.load("batch_booster_OC_test.txt", ds_test);
  cout << "testing batch_booster_OC again...";
  accuracy = 0;
  for (size_t i = 0; i < ds_test.size(); ++i)
    if (booster.predict(ds_test[i]) == ds_test.finite(i,class_var_index))
      ++accuracy;
  accuracy /= ds_test.size();
  cout << "Test accuracy on " << ntest << " test points: " << accuracy << endl;

}
