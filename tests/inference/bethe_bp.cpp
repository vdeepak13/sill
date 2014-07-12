#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/inference/asynchronous_bethe_bp.hpp>
#include <sill/inference/residual_bethe_bp.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>
#include <sill/model/decomposable.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace sill;

  if (argc < 4) {
    cout << "Usage: belief_propagation m n niters engine [eta]" << endl;
    return 1;
  }

  size_t m         = boost::lexical_cast<size_t>(argv[1]);
  size_t n         = boost::lexical_cast<size_t>(argv[2]);
  size_t niters    = boost::lexical_cast<size_t>(argv[3]);
  size_t engine_id = (argc <= 4) ? 1 : boost::lexical_cast<size_t>(argv[4]);
  double eta       = (argc <= 5) ? 1 : boost::lexical_cast<double>(argv[5]);

  boost::mt19937 rng;
  universe u;

  finite_var_vector variables = u.new_finite_variables(m*n, 2);
  cout << "Generating random model" << endl;
  pairwise_markov_network<table_factor> mn;
  make_grid_graph(variables, m, n, mn);
  random_ising_model(0.5, 1, mn, rng);
  if(m<10) cout << mn; 
  
  bethe_bp<table_factor>* engine;

  switch(engine_id) {
  case 1: engine = new asynchronous_bethe_bp<table_factor>(mn); break;
  case 2: engine = new residual_bethe_bp<table_factor>(mn); break;
  default: assert(false);
  }

  for(size_t i = 0; i < niters; i++) {
    double error = engine->iterate(eta);
    cout << "Iteration " << i << ": residual " << error << endl;
  }

  // Compute the exact answer and compare
  decomposable<table_factor> dm;
  dm *= mn.factors();
  double total_error = 0;
  for(size_t i = 0; i < variables.size(); i++) {
    table_factor exact = dm.marginal(make_domain(variables[i]));
    table_factor approx = engine->belief(make_domain(variables[i]));
    double error = norm_inf(exact, approx);
    total_error += error;
    cout << "Variable " << i << ": error " << error << endl;
  }
  cout << "Average error: " << total_error / variables.size() << endl;
}
