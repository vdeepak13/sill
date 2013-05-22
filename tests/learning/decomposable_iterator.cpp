#include <iostream>
#include <iterator>
#include <sill/data.hpp>
#include <sill/assignment.hpp>
#include <sill/variable.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/learning/decomposable_iterator.hpp>

using namespace sill;

// Test of decomposable iterators.
int main() {

  // load data
  universe universe;
  string directory("data");
  string filename("jtest.sum");
  data_t data(filename, directory, SYMBOLIC, universe);
  std::cout << data << std::endl;

  typedef sill::table_factor<double, dense_table> factor_t;
  typedef std::vector<factor_t> factor_vec_t;
  // create initial decomposable model
  decomposable<factor_t> model;
  // Create a factor for each variable X_i and E_i.
  domain X = data.variables();
  factor_vec_t factor_ptr_vec;
  domain::const_iterator d_it1 = X.begin();
  domain::const_iterator d_it2 = X.begin();
  domain::const_iterator d_it3 = X.begin();
  ++d_it2;
  ++d_it3;
  ++d_it3;
  while(d_it3 != X.end()) {
    domain tmp_domain
      = make_domain(*d_it1, *d_it2, *d_it3);
    factor_ptr_vec.push_back(data.marginal<factor_t>(tmp_domain));
    ++d_it1;
    ++d_it2;
    ++d_it3;
    if (d_it3 == X.end())
      break;
    ++d_it1;
    ++d_it2;
    ++d_it3;
  }
  model.multiply_in(factor_ptr_vec.begin(), factor_ptr_vec.end());

  std::cout << "initial model: " << model << std::endl;

  std::cout << "iterator over LOCAL_PUSH_BACK_SUBTREE:" << std::endl;

  std::vector<structure_step_enum>
    steps(1, STEP_LOCAL_PUSH_BACK_SUBTREE);
  param_method_enum param_method = PARAM_LOCAL_MLE;
  domain::size_type max_clique_size = 3;
  decomposable_iterator<factor_t> it1
    (model, data, steps, param_method, max_clique_size);
  decomposable_iterator<factor_t> end;

  while (it1 != end) {
    std::cout << *it1 << std::endl;
    ++it1;
  }

  std::cout << "iterator over LOCAL_COMBINE_NODES:" << std::endl;

  steps = std::vector<structure_step_enum>
    (1, STEP_LOCAL_COMBINE_NODES);
  param_method = PARAM_LOCAL_MLE;
  max_clique_size = 5;

  decomposable_iterator<factor_t> it2
    (model, data, steps, param_method, max_clique_size);

  while (it2 != end) {
    model.check_validity();
    std::cout << *it2 << std::endl;
    ++it2;
  }

  model.check_validity();

  std::cout << "iterator over LOCAL_EDGE_2_NODE:" << std::endl;

  steps = std::vector<structure_step_enum>
    (1, STEP_LOCAL_EDGE_2_NODE);
  param_method = PARAM_LOCAL_MLE;
  max_clique_size = 3;
  decomposable_iterator<factor_t> it3
    (model, data, steps, param_method, max_clique_size);

  model.check_validity();

  while (it3 != end) {
    std::cout << *it3 << std::endl;
    ++it3;
  }

  std::cout << "iterator over LONG_MOVE_LEAF:" << std::endl;

  steps = std::vector<structure_step_enum>
    (1, STEP_LONG_MOVE_LEAF);
  param_method = PARAM_LOCAL_MLE;
  max_clique_size = 3;
  decomposable_iterator<factor_t> it4
    (model, data, steps, param_method, max_clique_size);

  model.check_validity();

  while (it4 != end) {
    std::cout << *it4 << std::endl;
    ++it4;
  }

  std::cout << "iterator over LONG_SWAP_VAR:" << std::endl;

  steps = std::vector<structure_step_enum>
    (1, STEP_LONG_SWAP_VAR);
  param_method = PARAM_LOCAL_MLE;
  max_clique_size = 3;
  decomposable_iterator<factor_t> it5
    (model, data, steps, param_method, max_clique_size);
  model.check_validity();
  while (it5 != end) {
    std::cout << *it5 << std::endl;
    ++it5;
  }

  std::cout << "iterator over INIT_MODEL_EMPTY/STAR:" << std::endl;

  std::vector<structure_initial_enum>
    model_types(1, INIT_MODEL_EMPTY);
  model_types.push_back(INIT_MODEL_STAR);
  max_clique_size = 5;
  decomposable_iterator<factor_t> m_it1
    (data, data.variables(), model_types,
     param_method, max_clique_size);

  while (m_it1 != end) {
    std::cout << *m_it1 << std::endl;
    ++m_it1;
  }

}
