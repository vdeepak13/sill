// Tess the basic update rule

#include <stdlib.h>
#include <iostream>
#include <sstream>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>


#include <prl/base/universe.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/model/factor_graph_model.hpp>

#include <prl/factor/random.hpp>
#include <prl/parallel/pthread_tools.hpp>


#include <prl/inference/parallel/basic_state_manager.hpp>
#include <prl/inference/parallel/basic_update_rule.hpp>


// This should come last
#include <prl/macros_def.hpp>

// Including the standard and the prl namespaces
using namespace std;
using namespace prl;

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
  boost::uniform_01<boost::mt19937, double> unif01(rng);
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
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);
    table_factor factor(arguments, 1.0);
    factor.normalize();
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<table_factor>(arguments, unif01));
  }
  // Add edge parameters
  for(size_t i = 0; i < x.size()-1; ++i) {
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);   arguments.insert(x[i+1]);
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<table_factor>(arguments, unif01));
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
