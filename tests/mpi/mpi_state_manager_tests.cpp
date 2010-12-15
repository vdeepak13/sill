
// This file tests the mpi_state_manager

#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>
#include <prl/factor/random.hpp>

#include <prl/mpi/mpi_wrapper.hpp>
#include <prl/mpi/mpilogger.hpp>
#include <prl/inference/parallel/mpi_state_manager.hpp>
#include <prl/inference/parallel/mpi_state_manager_protocol.hpp>
#include <prl/base/universe.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/model/factor_graph_model.hpp>
#include <prl/inference/parallel/residual_splash_engine.hpp>


using namespace std;
using namespace prl;

void create_test_graph(universe &u, factor_graph_model<tablef> &fg) {
  boost::mt19937 rng;
  boost::uniform_01<boost::mt19937, double> unif01(rng);

  // Create some variables and factors
  std::vector<finite_variable*> x(100);
  for(size_t i = 0; i < x.size(); ++i) {
    char c[16];
    sprintf(c,"%d",i);
    x[i] = u.new_finite_variable(c, 2);
  }

  double bound = 0;
  size_t splash_size = 0;

  // Create some unary factors
  for(size_t i = 0; i < x.size(); ++i) {
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<tablef>(arguments, rng));
  }
  
  // For every two variables in a chain create a factor
  for(size_t i = 0; i < x.size() - 1; ++i) {
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);   arguments.insert(x[i+1]);
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<tablef>(arguments, rng));
  }

  basic_state_manager<tablef> manager(&fg, true);
}

typedef residual_splash_engine<tablef,mpi_state_manager<tablef> > 
    residual_splash_type;


int main(int argc, char** argv) {
  universe u;
  mpi_post_office po;
  factor_graph_model<tablef> fg;
    
  mpi_state_manager<tablef> *state; 
  
  std::cout << "Name:  " << po.name() << std::endl;

  std::cout << "Registering handlers." << std::endl;
  if (po.id() == 0) {
    create_test_graph(u, fg);
    state = new mpi_state_manager<tablef>(po,     // post office
                                          &fg,    // factor graph
                                          1.0E-5, // epsilon
                                          101);   // max vertices per node
  }
  else {
    state = new mpi_state_manager<tablef>(po);
  }

  po.start();
  std::cout << "MPI Started!\n";
  state->start();
  
  residual_splash_type engine(*state,
                            1,
                            25, 
                            1.0E-5,
                            0.4);
  std::cout << "FIN!\n";
  MPI::COMM_WORLD.Barrier();
  if (po.id() == 0) {
    state->collect_beliefs();
    std::cout << "Received Beliefs\n";
  }
  MPI::COMM_WORLD.Barrier();
  po.stopAll();
  po.wait();
  
}
