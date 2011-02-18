#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/discriminative/all_pairs_batch.hpp>
#include <sill/learning/discriminative/batch_booster.hpp>
#include <sill/learning/discriminative/boosters.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/filtering_booster.hpp>
#include <sill/learning/discriminative/stump.hpp>
#include <sill/learning/learn_factor.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  size_t ntrain = 1000;
  size_t ntest = 1000;
  size_t niterations = 25;
  double random_seed = 340891347;

  // Load a dataset to work with
  universe u;
  boost::shared_ptr<vector_dataset> ds_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>
    ("../../../../tests/data/uci/letter.sum", u, ntrain + ntest);
  //  ds_.randomize(29875123);
  dataset_view ds_train(*ds_ptr);
  ds_train.set_record_range(0, ntrain);
  dataset_view ds_test(*ds_ptr);
  ds_test.set_record_range(ntrain, ntrain + ntest);
  dataset_statistics stats_train(ds_train);

  finite_variable* class_var = ds_train.finite_class_variables().front();
  assert(class_var->size() == 26);
  size_t class_var_index = ds_train.record_index(class_var);
  finite_variable* binary_var = u.new_finite_variable(2);

  cout << "Marginal (from training set) over class variable:\n"
       << learn_marginal<table_factor>(make_domain(class_var), ds_train)
       << endl;

  typedef stump<> stump_type;
  boost::shared_ptr<stump_type> wl_ptr(new stump_type());
  typedef batch_booster<boosting::adaboost>
    batch_booster_type;
  batch_booster_parameters batch_booster_params;
  batch_booster_params.init_iterations = niterations;
  batch_booster_params.weak_learner = wl_ptr;
  boost::shared_ptr<batch_booster_type> bb_ptr
    (new batch_booster_type(batch_booster_params));

  typedef all_pairs_batch all_pairs_type;
  all_pairs_batch_parameters all_pairs_params;
  all_pairs_params.binary_label = binary_var;
  all_pairs_params.random_seed = random_seed;
  all_pairs_params.base_learner = bb_ptr;
  cout << "Beginning training of all_pairs_batch" << endl;
  all_pairs_type all_pairs(stats_train, all_pairs_params);
  cout << "Finished training all_pairs_batch" << endl;

  double accuracy = 0;
  for (size_t i = 0; i < ds_test.size(); ++i)
    if (all_pairs.predict(ds_test[i]) == ds_test.finite(i,class_var_index))
      ++accuracy;
  accuracy /= ds_test.size();
  cout << "Test accuracy on " << ntest << " test points: " << accuracy << endl;

  cout << "Saving all_pairs_batch...";
  all_pairs.save("all_pairs_batch_test.txt");
  cout << "loading all_pairs_batch...";
  all_pairs.load("all_pairs_batch_test.txt", ds_test);
all_pairs.save("all_pairs_batch_test2.txt");
  cout << "testing all_pairs_batch again...";
  accuracy = 0;
  for (size_t i = 0; i < ds_test.size(); ++i)
    if (all_pairs.predict(ds_test[i]) == ds_test.finite(i,class_var_index))
      ++accuracy;
  accuracy /= ds_test.size();
  cout << "Test accuracy on " << ntest << " test points: " << accuracy << endl;

}
