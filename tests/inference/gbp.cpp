#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <prl/inference/asynchronous_gbp_pc.hpp>
#include <prl/inference/kikuchi.hpp>
#include <prl/inference/bethe.hpp>

#include <prl/factor/table_factor.hpp>
#include <prl/graph/grid_graphs.hpp>
#include <prl/model/markov_network.hpp>
#include <prl/model/random.hpp>
#include <prl/model/decomposable.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;
  
  if (argc < 4) {
    cout << "Usage: gbp m n niters region_graph [eta]" << endl;
    return 1;
  }
  
  size_t m         = boost::lexical_cast<size_t>(argv[1]);
  size_t n         = boost::lexical_cast<size_t>(argv[2]);
  size_t niters    = boost::lexical_cast<size_t>(argv[3]);
  size_t rg_type   = (argc <= 4) ? 1 : boost::lexical_cast<size_t>(argv[4]);
  double eta       = (argc <= 5) ? 1 : boost::lexical_cast<double>(argv[5]);
  
  boost::mt19937 rng;
  universe u;
                      
  finite_var_vector variables = u.new_finite_variables(m*n, 2);
  cout << "Generating random model" << endl;
  pairwise_markov_network<table_factor> mn;
  boost::multi_array<finite_variable*, 2> vars = 
    make_grid_graph(m, n, mn, variables);
  random_ising_model(0.5, 1, mn, rng);
  if (m < 10) cout << mn;
  
  // Create the region graph with root clusters over 2x2 variables
  region_graph<finite_variable*, table_factor> rg;
  switch (rg_type) {
  case 1: { // Bethe
    std::vector<finite_domain> clusters;
    foreach(const table_factor& f, mn.factors())
      clusters.push_back(f.arguments());
    bethe(clusters, rg);
    break;
  }
  case 2: { // Kikuchi 
    std::vector<finite_domain> root_clusters;
    for(size_t i = 0; i < m-1; i++)
      for(size_t j = 0; j < n-1; j++) {
        finite_domain cluster = 
          make_domain(vars[i][j], vars[i+1][j], vars[i][j+1], vars[i+1][j+1]);
        root_clusters.push_back(cluster);
      }
    kikuchi(root_clusters, rg);
    break;
  }
  default:
    assert(false);
  }

  if (m < 10) cout << rg;

  gbp_pc<table_factor>* engine = new asynchronous_gbp_pc<table_factor>(rg); 
  engine->initialize_factors(mn);

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

  // Check if the edge marginals agree
  double max_error = 0;
  foreach(directed_edge<size_t> e, rg.edges()) {
    table_factor bel_s = engine->belief(e.source()).marginal(rg.cluster(e.target()));
    table_factor bel_t = engine->belief(e.target());
    double error = norm_inf(bel_s, bel_t);
    if (error > max_error) max_error = error;
  }
  cout << "Belief consistency: " << max_error << endl;
}
