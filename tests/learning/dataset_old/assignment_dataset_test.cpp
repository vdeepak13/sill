#include <iostream>
#include <string>

#include <sill/argument/universe.hpp>
#include <sill/learning/dataset_old/assignment_dataset.hpp>
//#include <sill/learning/dataset_old/concepts.hpp>
#include <sill/learning/dataset_old/data_loader.hpp>
#include <sill/factor/table_factor.hpp>

#include <sill/macros_def.hpp>

/**
 * \file assignment_dataset.cpp Test of the assignment dataset class.
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  std::string filename = argc > 1 ? argv[1] : "../../../../tests/data/spam.sum";

  universe u;

  //  concept_assert((sill::MutableDataset<assignment_dataset>));

  boost::shared_ptr<assignment_dataset<> > data_ptr =
    data_loader::load_symbolic_dataset<assignment_dataset<> >(filename, u);

  // Print the data
  cout << *data_ptr << endl;

  // Print the data as a sequence of assignments
  foreach(const record<>& rec, data_ptr->records()) 
    cout << rec.assignment() << endl;

  // Iterate over the data, computing the mean of the vector data
  vec v(zeros<vec>(data_ptr->vector_dim()));
  foreach(const record<>& record, data_ptr->records())
    v += record.vector();

  cout << (v / double(data_ptr->size())) << endl;

  return 0;
}
