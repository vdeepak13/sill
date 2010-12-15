#include <iostream>
#include <string>

#include <prl/variable.hpp>
#include <prl/assignment.hpp>
#include <prl/learning/dataset/array_dataset.hpp>
#include <prl/learning/dataset/concepts.hpp>
#include <prl/learning/dataset/data_conversions.hpp>
#include <prl/factor/table_factor.hpp>

#include <prl/macros_def.hpp>

/**
 * \file array_dataset.cpp Test of the array_dataset class.
 */
int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  std::string filename = argc > 1 ? argv[1] : "../../../../tests/data/spam.txt";

  universe u;
  vector_var_vector vv(u.new_vector_variables(10, 1));
  finite_var_vector fv;
  fv.push_back(u.new_finite_variable(2));
  std::vector<datasource::variable_type_enum>
    var_type_order(10, datasource::VECTOR_VAR_TYPE);
  var_type_order.push_back(datasource::FINITE_VAR_TYPE);

  concept_assert((prl::MutableDataset<array_dataset>));

  array_dataset data(*(load_plain<array_dataset>
                       (filename, fv, vv, var_type_order)));

  // Print the data
  cout << data << endl;

  // Print the data as a sequence of assignments
  foreach(const record& rec, data.records()) 
    cout << rec.assignment() << endl;

  // Iterate over the data, computing the mean of the vector data
  vec v(data.vector_dim(), 0);
  foreach(const record& record, data.records())
    v += record.vector();

  cout << (v / double(data.size())) << endl;

  return 0;
}
