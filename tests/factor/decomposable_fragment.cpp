#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/factor/decomposable_fragment.hpp>
#include <sill/graph/grid_graphs.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>
#include <sill/inference/junction_tree_inference.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char** argv) {
  using namespace sill;
  using namespace std;

  boost::mt19937 rng;

  assert(argc==3);

  // Loads a small tree-width model
  size_t m = boost::lexical_cast<size_t>(argv[1]);
  size_t n = boost::lexical_cast<size_t>(argv[2]);

  universe u;

  cout << "Generating random model" << endl;
  pairwise_markov_network< table_factor > mn;
  finite_var_vector variables = u.new_finite_variables(m*n, 2);
  make_grid_graph(m, n, mn, variables);
  random_ising_model(mn, rng);
  if (m<10) cout << mn;

  cout << "Marginals using junction tree inference: " << endl;
  shafer_shenoy<table_factor> ss(mn);
  ss.calibrate();
  ss.normalize();
  cout << ss.clique_beliefs() << endl;

  cout << "The corresponding decomposable_fragment: " << endl;
  decomposable_fragment<table_factor> df(ss.clique_beliefs());
  cout << df << endl;

  finite_domain v03 = make_domain(variables[0], variables[3]);
  decomposable_fragment<table_factor> df03 = df.marginal(v03);
  cout << "Marginal over " << v03 << endl;
  cout << df03 << endl;
  cout << df03.flatten() << endl;

  decomposable<table_factor> dm;
  dm *= mn.factors();
  cout << "Marginal over " << v03 << " using decomposable model: " << endl;
  cout << dm.marginal(v03) << endl;
}
