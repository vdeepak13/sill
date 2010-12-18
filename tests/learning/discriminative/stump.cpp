#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/statistics.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/stump.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Create a dataset to work with
  universe u;
  syn_oracle_knorm::parameters oracle_params;
  oracle_params.radius = 1;
  oracle_params.std_dev = 2.5;
  syn_oracle_knorm knorm(create_syn_oracle_knorm(2,3,u,oracle_params));
  finite_variable* class_var = knorm.finite_class_variables().front();
  cout << knorm;

  size_t ntrain = 100;
  size_t ntest = 40;

  boost::shared_ptr<vector_dataset> ds_train_ptr
    = oracle2dataset<vector_dataset>(knorm, ntrain);
  vector_dataset& ds_train = *ds_train_ptr;
  boost::shared_ptr<vector_dataset> ds_test_ptr
    = oracle2dataset<vector_dataset>(knorm, ntest);
  vector_dataset& ds_test = *ds_test_ptr;
  statistics stats(ds_train);

  stump<> s(stats);

  /*
  // Can also construct stumps this way:
  boost::shared_ptr<stump> s_ptr(s.createB(stats));
  boost::shared_ptr<multiclass_classifier> s_ptr2(s.createMC(stats));
  */

  cout << "Trained stump on " << ntrain << " examples\n"
       << "now testing on " << ntest << " examples" << endl;

  size_t nright = 0;
  foreach(const record& example, ds_test.records()) {
    const assignment& a = example.assignment();
    size_t predicted = s.predict(a);
    size_t truth = safe_get(a.finite(), class_var);
    if (predicted == truth)
      nright++;
  }
  cout << "Test accuracy = " << ((double)(nright) / ntest) << endl;

  // -- begin temp
  nright = 0;
  dataset::record_iterator end = ds_test.end();
  for (dataset::record_iterator it = ds_test.begin();
       it != end; ++it) {
    const record& example = *it;
    const assignment& a = example.assignment();
    size_t predicted = s.predict(a);
    size_t truth = safe_get(a.finite(), class_var);
    if (predicted == truth)
      nright++;
  }
  cout << "Test accuracy = " << ((double)(nright) / ntest) << endl;
  // -- end temp

  cout << "Saving stump...";
  s.save("stump_test.txt");
  cout << "loading stump...";
  s.load("stump_test.txt", ds_test);
  cout << "testing stump again...";
  nright = 0;
  foreach(const record& example, ds_test.records()) {
    const assignment& a = example.assignment();
    size_t predicted = s.predict(a);
    size_t truth = safe_get(a.finite(), class_var);
    if (predicted == truth)
      nright++;
  }
  cout << "Test accuracy = " << ((double)(nright) / ntest) << endl;

}
