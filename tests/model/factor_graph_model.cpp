#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>


#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/random.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

typedef factor_graph_model<table_factor> factor_gm_type;
typedef factor_gm_type::factor_type factor_type;
typedef factor_gm_type::variable_type variable_type;
typedef factor_gm_type::vertex_type vertex_type;

/**
 * \file factor_graph_model.cpp Factor Graph Model test
 */
int main() {
  

  using boost::array;

  // Random number generator
  boost::mt19937 rng;
  boost::uniform_01<boost::mt19937, double> unif01(rng);

  // Create a universe.
  universe u;
 
  // Create an empty factor graph
  factor_gm_type fg;
  std::cout << "Empty factory graph: " << std::endl;
  fg.print(std::cout);

  // Create some variables and factors
  std::vector<variable_type*> x(10);
  for(size_t i = 0; i < x.size(); ++i) 
    x[i] = u.new_finite_variable("Variable: " + to_string(i), 2);

  // Create some unary factors
  for(size_t i = 0; i < x.size(); ++i) {
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<factor_type>(arguments, unif01));
  }
  
  // For every two variables in a chain create a factor
  for(size_t i = 0; i < x.size() - 1; ++i) {
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);   arguments.insert(x[i+1]);
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<factor_type>(arguments, unif01));
  }
  std::ofstream out("factorgraph.xml");
  oarchive oa(out);
  oa << fg;
  out.close();
  
  
  std::cout << "Print the Factor graph model:" << std::endl;
  fg.print(std::cout);

  std::cout << "Print all the factors associated with each node:" << std::endl;
  foreach(finite_variable* var, fg.arguments()) {
    std::cout << "============================================================" 
              << std::endl
              << "Variable: " << var <<std::endl
              << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
              << "Neighbors: " << std::endl;
    vertex_type vert = fg.to_vertex(var);
    foreach(const vertex_type& v, fg.neighbors(vert)  )
      std::cout << v.factor() << std::endl; 
 
  }
}
