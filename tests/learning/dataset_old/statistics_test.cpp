#include <iostream>
#include <string>

#include <sill/argument/universe.hpp>
#include <sill/learning/dataset_old/assignment_dataset.hpp>
#include <sill/learning/dataset_old/data_conversions.hpp>
#include <sill/learning/dataset_old/dataset_statistics.hpp>
#include <sill/learning/dataset_old/syn_oracle_knorm.hpp>

#include <sill/macros_def.hpp>

/**
 * \file dataset.cpp Test of the dataset_statistics class using
 *                   assignment_dataset.
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Create a dataset with vector variables to work with
  universe u;
  syn_oracle_knorm knorm(create_syn_oracle_knorm(2,3,u));
  assignment_dataset<> ds;
  oracle2dataset(knorm, 10, ds);
  cout << "Dataset:\n" << ds << endl;

  // Print the data, along with its order dataset_statistics.
  cout << "Computing order stats and, for each vector variable, printing "
       << "the record values in increasing order:\n" << endl;
  dataset_statistics<> stats(ds);
  stats.compute_order_stats();
  const std::vector<std::vector<size_t> >& order_stats = stats.order_stats();
  const domain& vector_list = ds.vector_list();
  size_t d = 0; // counts from 0 to ds.vector_dim() - 1
  for (size_t j = 0; j < ds.num_vector(); j++) {
    for (size_t k = 0; k < vector_list[j]->size(); k++) {
      cout << "Variable " << vector_list[j] << ", element " << k << ":" << endl;
      for (size_t i = 0; i < ds.size(); i++)
        cout << ds.vector(order_stats[d][i],d) << ", ";
      cout << endl;
      d++;
    }
  }

  // Create a dataset with finite variables

  return 0;
}
