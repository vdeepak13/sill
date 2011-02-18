#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/filtering_booster_OC.hpp>
#include <sill/learning/discriminative/boosters.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/stump.hpp>
#include <sill/learning/learn_factor.hpp>

#include <sill/macros_def.hpp>

/**
 * Test FilterBoost OC on letter dataset from UCI ML Repository.
 * This performs about the same as the (approximately) identical test done in
 *  Schapire 1997 (ICML).
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  size_t ntrain = 5000;
  size_t ntest = 1000;
  size_t niterations = 20;
  double random_seed = 34089137;
  size_t m_t = 500;
  size_t n_t = 500;

  // Load a dataset to work with
  universe u;
  boost::shared_ptr<vector_dataset> ds_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>
    ("/Users/jbradley/data/uci/letter.sum", u, ntrain + ntest);
  
  ds_ptr->randomize(2987513);
  dataset_view ds_train(*ds_ptr);
  ds_train.set_record_range(0, ntrain);
  dataset_view ds_test(*ds_ptr);
  ds_test.set_record_range(ntrain, ntrain + ntest);
  dataset_statistics stats_train(ds_train);

  finite_variable* class_var = ds_train.finite_class_variables().front();
  size_t class_var_index = ds_train.record_index(class_var);
  finite_variable* binary_var = u.new_finite_variable(2);

  cout << "Marginal (from training set) over class variable:\n"
       << learn_marginal<table_factor>(make_domain(class_var), ds_train)
       << endl;

  typedef stump<> wl_type;
  boost::shared_ptr<wl_type> wl_ptr(new wl_type());
  typedef filtering_booster_OC<boosting::filterboost> filtering_booster_type;
  filtering_booster_OC_parameters filtering_booster_params;
  filtering_booster_params.init_iterations = niterations;
  filtering_booster_params.random_seed = random_seed;
  filtering_booster_params.binary_label = binary_var;
  filtering_booster_params.scale_m_t = true;
  filtering_booster_params.scale_n_t = true;
  filtering_booster_params.m_t = m_t;
  filtering_booster_params.n_t = n_t;
  filtering_booster_params.weak_learner = wl_ptr;

  cout << "Running FilterBoost.OC..." << endl;
  filtering_booster_type booster(stats_train, filtering_booster_params);
  std::vector<double> accuracies(booster.test_accuracies(ds_test));
  cout << "Iter\t Test accuracy on " << ntest << " test points" << endl;
  for (size_t t = 0; t < accuracies.size(); ++t)
    cout << t << "\t" << accuracies[t] << endl;

  cout << "Saving filtering_booster_OC...";
  booster.save("filtering_booster_OC_test.txt");
  cout << "loading filtering_booster_OC...";
  booster.load("filtering_booster_OC_test.txt", ds_test);
  cout << "testing filtering_booster_OC again..." << endl;
  accuracies = booster.test_accuracies(ds_test);
  cout << "Iter\t Test accuracy on " << ntest << " test points" << endl;
  for (size_t t = 0; t < accuracies.size(); ++t)
    cout << t << "\t" << accuracies[t] << endl;

}
