// Tess the basic update rule

#include <stdlib.h>
#include <iostream>
#include <sstream>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/argument/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/parallel/pthread_tools.hpp>

#include <sill/inference/parallel/basic_state_manager.hpp>
#include <sill/inference/parallel/basic_update_rule.hpp>

// This should come last
#include <sill/macros_def.hpp>

// Including the standard and the prl namespaces
using namespace std;
using namespace sill;

typedef factor_graph_model<table_factor> factor_graph_model_type;
typedef basic_state_manager<table_factor> basic_state_manager_type;
typedef basic_update_rule<basic_state_manager_type> basic_update_rule_type;
typedef factor_graph_model_type::variable_type variable_type;
typedef factor_graph_model_type::vertex_type vertex_type;
typedef basic_state_manager_type::belief_type belief_type;
typedef basic_state_manager_type::message_type message_type;

/*
// This has been moved to base/stl_util.hpp (Joseph B.)
template<typename T>
string to_string(T x) {
  stringstream s;
  s << x;
  return s.str();
}
*/

int main(int argc, char* argv[]) {
  cout << "==================================================================="
       << endl
       << "Initializing basic_update_rule tests :" << endl;

  cout << "-- Creating the Universe : ";
  universe u; 
  cout << "Done!" << endl;
  cout << "-- Initializing random number genrators : ";
  boost::mt19937 rng;
  uniform_factor_generator gen;
  cout << "Done!" << endl;

  cout << "-- Creating variables : ";
  std::vector<variable_type*> x(10);
  for(size_t i = 0; i < x.size(); ++i) 
    x[i] = u.new_finite_variable("Var:" + to_string(i), 2);
  cout << "Done!" << endl;

  cout << "-- Creating Factor Graph : ";
  factor_graph_model_type fg;  

  // Add vertex parameters
  for(size_t i = 0; i < x.size(); ++i) {
    finite_domain arguments;
    arguments.insert(x[i]);
    fg.add_factor(gen(arguments, rng));
  }

  // Add edge parameters
  for(size_t i = 0; i < x.size()-1; ++i) {
    finite_domain arguments;  
    arguments.insert(x[i]);
    arguments.insert(x[i+1]);
    fg.add_factor(gen(arguments, rng));
  }
  cout << "Done!" << endl;


  cout << "-- Creating basic state manager : ";
  basic_state_manager_type state_manager(&fg);
  cout << "Done!" << endl;

  cout << "-- Creating basic update rule : ";
  basic_update_rule_type update_rule;
  cout << "Done!" << endl;

  cout << "Finished!  Beginning tests!" << endl;



  cout << "==================================================================="
       << endl
       << "Testing update belief function:" << endl;  
  cout << "-- Running update on all vertices" << endl;
  foreach(const vertex_type& v , fg.vertices()) {
    update_rule.update(v, state_manager);
  } 
  cout << "Done!" << endl;

  return EXIT_SUCCESS;
}
