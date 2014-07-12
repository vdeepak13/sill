#include <sill/graph/grid_graph.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>
#include <sill/inference/belief_propagation.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/dense_table.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <vector>
#include <iostream>

boost::mt19937 rng;

int main(int argc, char* argv[])
{
  using namespace sill;
  using std::cout;
  using std::endl;
  typedef pairwise_markov_network<table_factor> mn_type;

  if (argc!=5 && argc!=6) {
    cout << "Usage: belief_propagation m n engine n_iterations [eta]" << endl;
    return 1;
  }

  size_t m         = boost::lexical_cast<size_t>(argv[1]);
  size_t n         = boost::lexical_cast<size_t>(argv[2]);
  size_t engine_id = boost::lexical_cast<size_t>(argv[3]);
  size_t niters    = boost::lexical_cast<size_t>(argv[4]);
  double eta       = (argc==5) ? 1 : boost::lexical_cast<double>(argv[5]);

  universe u;
  finite_var_vector variables = u.new_finite_variables(m*n, 2);

  cout << "Generating random model" << endl;
  mn_type mn;
  make_grid_graph(variables, m, n, mn);
  //mn.extend_domains();
  //randomize_factors(mn, rng);
  random_ising_model(mn, rng);
  if(m<10) cout << mn;

  cout << "Running GBP" << endl;
  boost::shared_ptr<  loopy_bp_engine<mn_type>  > p_engine;
  switch(engine_id) {
  case 0: p_engine.reset(new asynchronous_loopy_bp<mn_type>(mn)); break;
  case 1: p_engine.reset(new residual_loopy_bp<mn_type>(mn)); break;
  default: assert(false);
  }
  p_engine->iterate(niters, eta);
  //if(m<10) cout << p_engine->node_beliefs() << endl;
  cout << p_engine->node_beliefs() << endl;
  cout << "Average L1 norm: " << p_engine->average_residual() << endl;
}
