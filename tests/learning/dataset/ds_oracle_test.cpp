#include <iostream>
#include <string>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/assignment_dataset.hpp>
#include <sill/learning/dataset/ds_oracle.hpp>
//#include <sill/learning/dataset/concepts.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/factor/table_factor.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  std::string filename = argc > 1 ? argv[1] : "../../../../tests/data/spam.sum";

  universe u;

  boost::shared_ptr<assignment_dataset<> > data_ptr
    = data_loader::load_symbolic_dataset<assignment_dataset<> >(filename, u);

  cout << "Dataset:\n" << *data_ptr << endl << endl;

  //  concept_assert((sill::Oracle<ds_oracle<> >));
  ds_oracle<> o(*data_ptr);
  cout << "Oracle using first 3 records:\n";
  for (size_t i = 0; i < 3; ++i) {
    o.next();
    cout << o.current().assignment() << endl;
  }
}
