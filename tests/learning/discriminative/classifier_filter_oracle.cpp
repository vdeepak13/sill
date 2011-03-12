#include <iostream>

#include <sill/learning/dataset/classifier_filter_oracle.hpp>
#include <sill/learning/dataset/ds_oracle.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/stump.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Create a dataset to work with
  universe u;
  syn_oracle_knorm::parameters oracle_params;
  oracle_params.radius = 1;
  oracle_params.std_dev = 3;
  syn_oracle_knorm knorm(create_syn_oracle_knorm(2,3,u,oracle_params));
//  finite_variable* class_var = knorm.finite_class_variables().front();
  cout << knorm;

  size_t ntrain = 100;
  size_t ntest = 40;

  boost::shared_ptr<vector_dataset<> > ds_train_ptr
    = oracle2dataset<vector_dataset<> >(knorm, ntrain);
  vector_dataset& ds_train = *ds_train_ptr;
  dataset_statistics<> stats(ds_train);

  stump<> s(stats);

  cout << "Trained stump on " << ntrain << " examples and "
       << "now making sure that a classifier_filter_oracle works on them."
       << endl;

  boost::shared_ptr<vector_dataset<> > ds_test_ptr
    = oracle2dataset<vector_dataset<> >(knorm, ntest);
  vector_dataset& ds_test = *ds_test_ptr;
  ds_oracle::parameters ds_o_params;
  ds_o_params.auto_reset = false;
  ds_oracle ds_o(ds_test, ds_o_params);

  classifier_filter_oracle::parameters params;
  params.label_value = 0;
  classifier_filter_oracle filter_o
    (ds_o, s, classifier_filter_oracle::IS_VALUE, params);

  // counts of how many 1's appear directly before each 0 label in the stump's
  // predictions about ds_test
  std::vector<size_t> counts;

  size_t i = 0;
  size_t count = 0;
  foreach(const record& example, ds_test.records()) {
    size_t predicted = s.predict(example);
    ++count;
    if (predicted == 0) {
      counts.push_back(count);
      count = 0;
    } else
    ++i;
  }

  i = 0;
  while (filter_o.next()) {
    assert(i < counts.size());
    assert(counts[i] == filter_o.count());
    ++i;
  }

  cout << "Yay it worked." << endl;

}
