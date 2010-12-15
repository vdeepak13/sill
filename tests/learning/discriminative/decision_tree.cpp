#include <iostream>

#include <prl/base/universe.hpp>
#include <prl/learning/dataset/data_conversions.hpp>
#include <prl/learning/dataset/statistics.hpp>
#include <prl/learning/dataset/syn_oracle_majority.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/discriminative/concepts.hpp>
#include <prl/learning/discriminative/decision_tree.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  // Create a dataset to work with
  universe u;
  syn_oracle_majority majority(create_syn_oracle_majority(100,u));
  finite_variable* class_var = majority.finite_class_variables().front();
  cout << majority;

  size_t ntrain = 500;
  size_t ntest = 500;

  boost::shared_ptr<vector_dataset> ds_train_ptr
    = oracle2dataset<vector_dataset>(majority, ntrain);
  vector_dataset& ds_train = *ds_train_ptr;
  boost::shared_ptr<vector_dataset> ds_test_ptr
    = oracle2dataset<vector_dataset>(majority, ntest);
  vector_dataset& ds_test = *ds_test_ptr;
  statistics stats(ds_train);

  decision_tree<> s(stats);

  cout << "Trained decision_tree on " << ntrain << " examples\n"
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

  cout << "Saving decision_tree...";
  s.save("decision_tree_test.txt");
  cout << "loading decision_tree...";
  s.load("decision_tree_test.txt", ds_test);
  cout << "testing decision_tree again...";
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
