#include <iostream>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/logistic_regression.hpp>
#include <sill/learning/discriminative/batch_booster.hpp>
#include <sill/learning/discriminative/boosters.hpp>
//#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/filtering_booster.hpp>
#include <sill/learning/discriminative/stump.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;
  boost::timer timer;

  // Create a dataset to work with
  universe u;
  syn_oracle_knorm::parameters oracle_params;
  oracle_params.radius = 1;
  oracle_params.std_dev = 2.5;
  oracle_params.random_seed = 489167211;
  syn_oracle_knorm knorm(create_syn_oracle_knorm(2,20,u,oracle_params));
  cout << knorm;

  size_t ntrain = 5000;
  size_t ntest = 5000;
  size_t niterations = 200;
  double random_seed = 340891347;

  vector_dataset<> ds_train;
  oracle2dataset(knorm, ntrain, ds_train);
  vector_dataset<> ds_test;
  oracle2dataset(knorm, ntest, ds_test);
  dataset_statistics<> stats(ds_train);

  typedef stump<> wl_type;
  boost::shared_ptr<binary_classifier<> > wl_ptr(new wl_type());
  std::vector<double> test_accuracies;

  typedef batch_booster<boosting::adaboost>
    batch_booster_type;
  if (0) {
  batch_booster_parameters batch_booster_params;
  batch_booster_params.random_seed = random_seed;
  batch_booster_params.weak_learner = wl_ptr;
  batch_booster_type batch_booster(stats, batch_booster_params);

  cout << "Training batch AdaBoost on " << ntrain << " examples for "
       << niterations << " iterations" << endl;
  timer.restart();
  for (size_t t = 0; t < niterations; t++)
    if (!(batch_booster.step()))
      break;
  cout << "  done in " << timer.elapsed() << " seconds; now testing on "
       << ntest << " examples" << endl;

  cout << "Iteration\tTest Accuracy" << endl;
  test_accuracies = batch_booster.test_accuracies(ds_test);
  for (size_t t = 0; t < test_accuracies.size(); t++)
    cout << t << "\t" << test_accuracies[t] << endl;
  }

  if (1) {
  batch_booster_parameters batch_booster_params;
  batch_booster_params.random_seed = random_seed;
  batch_booster_params.resampling = 500;
  batch_booster_params.weak_learner = wl_ptr;
  batch_booster_type batch_booster(stats, batch_booster_params);

  cout << "Training batch AdaBoost with resampling on " << ntrain
       << " examples for " << niterations << " iterations" << endl;
  timer.restart();
  for (size_t t = 0; t < niterations; t++) {
    if (!(batch_booster.step()))
      break;
  }
  cout << "  done in " << timer.elapsed() << " seconds; now testing on "
       << ntest << " examples" << endl;

  cout << "Iteration\tTest Accuracy" << endl;
  test_accuracies = batch_booster.test_accuracies(ds_test);
  for (size_t t = 0; t < test_accuracies.size(); t++)
    cout << t << "\t" << test_accuracies[t] << endl;

  cout << "Saving batch_booster...";
  batch_booster.save("batch_booster_test.txt");
  cout << "loading batch_booster...";
  batch_booster.load("batch_booster_test.txt", ds_test);
  cout << "testing batch_booster again...\n";
  cout << "Iteration\tTest Accuracy" << endl;
  test_accuracies = batch_booster.test_accuracies(ds_test);
  for (size_t t = 0; t < test_accuracies.size(); t++)
    cout << t << "\t" << test_accuracies[t] << endl;

  cout << "\nTiming testing, averaged over 1000 iterations: ";
  timer.restart();
  for (size_t t = 0; t < 1000; ++t) {
    test_accuracies = batch_booster.test_accuracies(ds_test);
    if (argc > 20)
      cout << test_accuracies[0];
  }
  cout << timer.elapsed() / 1000 << " seconds" << endl;

  }

  typedef filtering_booster<boosting::adaboost> filtering_booster_type;
  if (0) {
  filtering_booster_parameters filtering_booster_params;
  filtering_booster_params.random_seed = random_seed;
  filtering_booster_params.weak_learner = wl_ptr;
  filtering_booster_type filtering_booster
    (knorm, std::numeric_limits<size_t>::max(), filtering_booster_params);

  cout << "Training filtering AdaBoost on " << ntrain << " examples for "
       << niterations << " iterations\n";
  timer.restart();
  for (size_t t = 0; t < niterations; t++)
    filtering_booster.step();
  cout << "  done in " << timer.elapsed() << " seconds; now testing on "
       << ntest << " examples" << endl;

  cout << "Iteration\tTest Accuracy" << endl;
  test_accuracies = filtering_booster.test_accuracies(ds_test);
  for (size_t t = 0; t < test_accuracies.size(); t++)
    cout << t << "\t" << test_accuracies[t] << endl;
  }

}
