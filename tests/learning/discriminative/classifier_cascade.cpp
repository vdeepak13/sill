#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/syn_oracle_majority.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/batch_booster.hpp>
#include <sill/learning/discriminative/boosters.hpp>
#include <sill/learning/discriminative/classifier_cascade.hpp>
#include <sill/learning/discriminative/stump.hpp>
#include <sill/learning/learn_factor.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Create a dataset to work with
  universe u;

  size_t ntrain = 30162;
  size_t ntest = 15060;
  boost::shared_ptr<vector_dataset> ds_train_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>
    ("/Users/jbradley/data/uci/adult/adult-train.sum", u, ntrain);
  vector_dataset& ds_train = *ds_train_ptr;
  boost::shared_ptr<vector_dataset> ds_test_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>
    ("/Users/jbradley/data/uci/adult/adult-test.sum",
     ds_train.datasource_info(), ntest);
  vector_dataset& ds_test = *ds_test_ptr;
  finite_variable* class_var = ds_train.finite_class_variables().front();
  size_t class_var_index = ds_train.record_index(class_var);
  
  cout << "Marginal over training set class variables:\n"
       << learn_marginal<table_factor>(make_domain(class_var), ds_train)
       << endl;
  cout << "Now splitting training set into positive, negative." << endl;
  std::vector<size_t> pos_indices, neg_indices;
  for (size_t i(0); i < ds_train.size(); ++i)
    if (ds_train.finite(i,class_var_index) == 1)
      pos_indices.push_back(i);
    else
      neg_indices.push_back(i);
  dataset_view ds_view_pos(ds_train);
  ds_view_pos.set_record_indices(pos_indices);
  dataset_view ds_view_neg(ds_train);
  ds_view_neg.set_record_indices(neg_indices);
  ds_oracle ds_pos_oracle(ds_view_pos);

  classifier_cascade_parameters params;
  std::vector<boost::shared_ptr<binary_classifier> > base_classifiers;
  boost::shared_ptr<stump<> > wl_ptr(new stump<>());
  batch_booster_parameters batch_booster_params;
  batch_booster_params.init_iterations = 5;
  batch_booster_params.resampling = 100;
  batch_booster_params.weak_learner = wl_ptr;
  base_classifiers.push_back
    (boost::shared_ptr<binary_classifier>
     (new batch_booster<boosting::adaboost>(batch_booster_params)));
  params.base_classifiers = base_classifiers;
  params.rare_class = 0;
  params.max_false_common_rate = .85;
  params.random_seed = 239852;
  classifier_cascade cascade(ds_view_neg, ds_pos_oracle, params);

  cout << "Training a few rounds:\n" << endl;

  for (size_t t(0); t < 5; ++t) {
    if (!cascade.step())
      break;
    cout << "Did round " << t << "\n"
         << "\t test accuracy = " << cascade.test_accuracy(ds_test) << endl;
  }

  /*
// This was copied from another test, and it should be modified to test
// save and load on classifier_cascade as well.
  cout << "Saving batch_log_reg...";
  lr.save("batch_log_reg_test.txt");
  cout << "loading batch_log_reg...";
  lr.load("batch_log_reg_test.txt", ds_test);
  cout << "testing batch_log_reg again...";

  cout << "Saved and loaded logistic regression model:\n"
       << "  " << lr << "\n"
       << "now testing on " << ntest << " examples" << endl;

  cout << "Test accuracy = " << lr.test_accuracy(ds_test) << endl;
  */
}
