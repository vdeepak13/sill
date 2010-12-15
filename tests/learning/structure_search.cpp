#include <iostream>
#include <prl/data.hpp>
#include <prl/variable.hpp>
#include <prl/assignment.hpp>
#include <prl/learning/structure_search.hpp>

using namespace prl;

// Test of the structure_search class.
int main() {

  universe u;
  string directory("data");
  string filename("jtest2.sum");
  data_t data(filename, directory, SYMBOLIC, u);
  std::cout << data << std::endl;

  typedef table_factor<double, dense_table> factor_t;

  domain vars = data.variables();

  domain X;
  assignment e;
  int i = vars.size() / 2;
  for (domain::const_iterator it = vars.begin();
       it != vars.end(); ++it) {
    if (i > 0) {
      X = X.plus(*it);
      --i;
    } else {
      e[*it] = 0;
    }
  }

  typedef structure_search<factor_t> struct_search_t;

  std::vector<structure_initial_enum> initial_structures;
  initial_structures.push_back(INIT_MODEL_EMPTY);
  std::vector<structure_step_enum> step_types;
  step_types.push_back(STEP_LOCAL_PUSH_BACK_SUBTREE);
  step_types.push_back(STEP_LOCAL_COMBINE_NODES);
  step_types.push_back(STEP_LOCAL_EDGE_2_NODE);
  step_types.push_back(STEP_LONG_MOVE_LEAF);
  structure_objective_enum which_objective = QS_COND_ENTROPY;
  param_method_enum param_method = PARAM_LOCAL_MLE;
  domain::size_type max_clique_size = 3;
  std::size_t num_steps = 0;

  struct_search_t struct_search
    (X, e, data, initial_structures, step_types, which_objective, param_method,
     max_clique_size, num_steps);

  std::cout << "Initial model has score " << struct_search.current_score()
            << ":" << std::endl << struct_search.current_model();

  // do some steps and print the scores
  for (int i = 0; i < 10; ++i) {
    if (struct_search.step())
      std::cout << "Model on step " << i << " has score "
                << struct_search.current_score() << ":" << std::endl
                << struct_search.current_model();
    else {
      std::cout << "Could not find a better model." << std::endl;
      break;
    }
  }

  // Test functions in decomposable and data_t for computing log likelihoods.
  double loglike = data.train_log_likelihood(struct_search.current_model());
  std::cout << "Training data log likelihood = " << loglike << std::endl;

}
