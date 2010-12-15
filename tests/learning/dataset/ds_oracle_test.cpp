#include <iostream>
#include <string>

#include <prl/base/universe.hpp>
#include <prl/learning/dataset/assignment_dataset.hpp>
#include <prl/learning/dataset/ds_oracle.hpp>
#include <prl/learning/dataset/concepts.hpp>
#include <prl/learning/dataset/data_loader.hpp>
#include <prl/factor/table_factor.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  std::string filename = argc > 1 ? argv[1] : "../../../../tests/data/spam.sum";

  universe u;

  boost::shared_ptr<assignment_dataset> data_ptr
    = data_loader::load_symbolic_dataset<assignment_dataset>(filename, u);

  cout << "Dataset:\n" << *data_ptr << endl << endl;

  concept_assert((prl::Oracle<ds_oracle>));
  ds_oracle o(*data_ptr);
  cout << "Oracle using first 3 records:\n";
  for (size_t i = 0; i < 3; ++i) {
    o.next();
    const record& r = o.current();
    cout << o.current().assignment() << endl;
  }
}
