#include <iostream>
#include <vector>

#include <boost/array.hpp>

#include <sill/base/universe.hpp>
#include <sill/range/concepts.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/copy_ptr.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  symbolic::parameters sym_params;
  sym_params.prefix = "a";
  sym_params.index_base = 1;
  sym_params.skiplines = 1;

  boost::array<size_t, 7> var_sizes = {{2, 2, 3, 3, 2, 2, 2}};
  finite_var_vector var_order;
  universe u;
  foreach(size_t size, var_sizes) {
    var_order.push_back(u.new_finite_variable(size));
  }

  symbolic_oracle::parameters params;
  params.record_limit = 10;
  params.auto_reset = true;
  sym_params.datasource_info.finite_seq = var_order;
  sym_params.data_filename = "../../../../tests/data/jtest.txt.symbolic";
  symbolic_oracle o(sym_params, params);
  while(o.next())
    std::cout << o.current().assignment() << std::endl;
  // TODO: FIX THIS so we can just call:
  //     std::cout << o.current();

  std::cout << std::endl;

  symbolic_oracle
    o2(*(data_loader::load_symbolic_oracle("../../../../tests/data/jtest.sum",
                                           u)));
  while(o2.next())
    std::cout << o2.current().assignment() << "\t" << o2.weight() << std::endl;
  std::cout << std::endl;

  symbolic_oracle o3(o2);
  o3.reset();
  o3.next();
  std::cout << o3.current().assignment() << "\n" << std::endl;

}
