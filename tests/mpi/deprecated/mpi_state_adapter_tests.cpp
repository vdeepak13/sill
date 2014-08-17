#include <string>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <boost/random/mersenne_twister.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>

#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/mpi/mpi_consensus.hpp>
#include <sill/inference/parallel/mpi_state_adapter.hpp>
#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>

using namespace std;
using namespace sill;

void create_test_graph(universe &u, factor_graph_model<tablef> &fg) {
  boost::mt19937 rng;
  uniform_factor_generator gen;

  // Create some variables and factors
  std::vector<finite_variable*> x(5);
  for(size_t i = 0; i < x.size(); ++i) {
    std::ostringstream strm;
    strm << i;
    x[i] = u.new_finite_variable(strm.str(), 2);
  }

  // Create some unary factors
  for(size_t i = 0; i < x.size(); ++i) {
    finite_domain arguments;  
    arguments.insert(x[i]);
    fg.add_factor(gen(arguments, rng));
  }
  
  // For every two variables in a chain create a factor
  for(size_t i = 0; i < x.size() - 1; ++i) {
    finite_domain arguments;  
    arguments.insert(x[i]);
    arguments.insert(x[i+1]);
    fg.add_factor(gen(arguments, rng));
  }

  fg.simplify();
}


int main(int argc, char** argv) {
  universe u;
  mpi_post_office po;
  factor_graph_model<tablef> fg;
  po.start();
  std::cout << "STARTED!" << std::endl;
  mpi_simple_consensus consensus;
  consensus.init(&po, 1E-5);

  mpi_state_adapter adapter(&po, &consensus);
  

  size_t splash_size = 0;
  double bound = 0;
  double alpha = 0;

  if(po.id() == 0) {
    std::cout << "I am process 0" << std::endl;
    splash_size = 100;
    bound = 1E-5;
    alpha = 0.4;
    po.mpi_bcast(splash_size);
    po.mpi_bcast(bound);
    po.mpi_bcast(alpha);
    

    create_test_graph(u, fg);
    adapter.init(fg);
    // Send a kill message
    po.stopAll();
  } else {
    po.mpi_bcast(splash_size);
    po.mpi_bcast(bound);
    po.mpi_bcast(alpha);

    std::cout << "I am process " << po.id() << std::endl;
    adapter.init();
  }


  std::cout << "Splash Size: " << splash_size << " "
            << "Bound: " << bound << " "
            << "alpha: " << alpha << std::endl;



  
  std::cout << "FIN!\n";
  po.wait();  

  return EXIT_SUCCESS;
}


