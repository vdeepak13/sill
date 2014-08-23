#include <iostream>
#include <vector>

#include <boost/array.hpp>

#include <sill/base/universe.hpp>
#include <sill/range/concepts.hpp>
#include <sill/learning/dataset_old/syn_oracle_knorm.hpp>
#include <sill/learning/dataset_old/syn_oracle_majority.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  universe u;
  vector_var_vector vector_list;
  std::vector<variable::variable_typenames> var_type_order;

  // Test knorm oracle
  std::cout << "Test knorm oracle\n" << std::endl;
  size_t nmeans = 5;
  for (size_t k = 0; k < nmeans; k++) {
    vector_list.push_back(u.new_vector_variable(1));
    var_type_order.push_back(variable::VECTOR_VARIABLE);
  }
  finite_variable* fv = u.new_finite_variable(nmeans);
  var_type_order.push_back(variable::FINITE_VARIABLE);
  syn_oracle_knorm knorm(vector_list, fv, var_type_order);
  for (size_t i = 0; i < 5; i++) {
    knorm.next();
    std::cout << knorm.current().assignment() << std::endl;
  }
  syn_oracle_knorm knorm2(create_syn_oracle_knorm(2,3,u));
  for (size_t i = 0; i < 5; i++) {
    knorm2.next();
    std::cout << knorm2.current().assignment() << std::endl;
  }

  // Test majority vote oracle
  syn_oracle_majority majority(create_syn_oracle_majority(9,u));
  std::cout << "\n" << majority << std::endl;
  for (size_t i = 0; i < 5; ++i) {
    majority.next();
    std::cout << majority.current().assignment() << std::endl;
  }

}
