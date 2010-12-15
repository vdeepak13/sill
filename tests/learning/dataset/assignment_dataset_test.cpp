#include <iostream>
#include <string>

#include <prl/base/universe.hpp>
#include <prl/learning/dataset/assignment_dataset.hpp>
#include <prl/learning/dataset/concepts.hpp>
#include <prl/learning/dataset/data_loader.hpp>
#include <prl/factor/table_factor.hpp>

#include <prl/macros_def.hpp>

/**
 * \file assignment_dataset.cpp Test of the assignment dataset class.
 */
int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  std::string filename = argc > 1 ? argv[1] : "../../../../tests/data/spam.sum";

  universe u;

  concept_assert((prl::MutableDataset<assignment_dataset>));

  boost::shared_ptr<assignment_dataset> data_ptr =
    data_loader::load_symbolic_dataset<assignment_dataset>(filename, u);

  // Print the data
  cout << *data_ptr << endl;

  // Print the data as a sequence of assignments
  foreach(const record& rec, data_ptr->records()) 
    cout << rec.assignment() << endl;

  // Iterate over the data, computing the mean of the vector data
  vec v(data_ptr->vector_dim(), 0);
  foreach(const record& record, data_ptr->records())
    v += record.vector();

  cout << (v / double(data_ptr->size())) << endl;

  return 0;
}
