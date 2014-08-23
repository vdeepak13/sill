#include <iostream>
#include <string>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
//#include <sill/learning/dataset_old/concepts.hpp>
#include <sill/learning/dataset_old/data_loader.hpp>
#include <sill/learning/dataset_old/data_conversions.hpp>
#include <sill/learning/dataset_old/syn_oracle_majority.hpp>
#include <sill/learning/dataset_old/vector_dataset.hpp>

#include <sill/macros_def.hpp>

/**
 * \file vector_dataset_old.cpp Test of the vector dataset class.
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  std::string filename = argc > 1 ? argv[1] : "../../../../tests/data/spam.sum";

  universe u;

  //  concept_assert((sill::MutableDataset<vector_dataset_old>));

  boost::shared_ptr<vector_dataset_old<> > data_ptr =
    data_loader::load_symbolic_dataset<vector_dataset_old<> >(filename, u);

  // Print the data
  cout << *data_ptr << endl;

  // Print the data as a sequence of assignments
  foreach(const record<>& r, data_ptr->records()) 
    cout << r.assignment() << endl;

  // Iterate over the data, computing the mean of the vector data
  vec v(zeros<vec>(data_ptr->vector_dim()));
  foreach(const record<>& r, data_ptr->records())
    v += r.vector();

  cout << (v / double(data_ptr->size())) << endl;

  // Split the data into 2 parts.
  vector_dataset_old<> spam1(data_ptr->datasource_info());
  vector_dataset_old<> spam2(data_ptr->datasource_info());
  for (size_t i(0); i < data_ptr->size() / 2; ++i) {
    spam2.insert(data_ptr->operator[](i));
  }
  for (size_t i(data_ptr->size() / 2); i < data_ptr->size(); ++i) {
    spam1.insert(data_ptr->operator[](i));
  }
  // Normalize part 1, and use that normalization for part 2.
  cout << "Testing normalization...\n"
       << "Original spam data 1:\n"
       << spam1 << "\n"
       << "Original spam data 2:\n"
       << spam2 << endl;
  std::pair<vec,vec> normalizer(spam1.normalize(spam1.vector_list()));
  spam2.normalize(normalizer.first, normalizer.second, spam2.vector_list());
  cout << "Normalized spam data 1:\n"
       << spam1 << "\n"
       << "Normalized spam data 2:\n"
       << spam2 << endl;

  // Test on finite data
  syn_oracle_majority majority(create_syn_oracle_majority(9,u));
  vector_dataset_old<> majority_ds;
  oracle2dataset(majority, 20, majority_ds);

  return 0;
}
