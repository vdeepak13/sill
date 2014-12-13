#define BOOST_TEST_MODULE mean_field_pairwise
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_matrix.hpp>
#include <sill/factor/probability_matrix.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/bipartite_graph.hpp>
#include <sill/graph/special/grid_graph.hpp>
#include <sill/inference/exact/junction_tree_inference.hpp>
#include <sill/inference/variational/mean_field_pairwise.hpp>
#include <sill/model/markov_network.hpp>

#include <boost/random/uniform_int_distribution.hpp>

#include <sill/macros_def.hpp>

BOOST_AUTO_TEST_CASE(test_convergence) {
  using namespace sill;

  size_t m = 8;
  size_t n = 5;
  size_t niters = 20;

  // Create a grid graph
  universe u;
  uniform_factor_generator gen;
  boost::mt19937 rng;
  pairwise_markov_network<canonical_matrix<> > model;
  std::vector<table_factor> factors;
  finite_var_vector vars = u.new_finite_variables(m * n, 2);
  arma::field<finite_variable*> grid = make_grid_graph(vars, m, n, model);
  
  // node potentials
  foreach (finite_variable* v, model.vertices()) {
    table_factor f = gen(make_domain(v), rng);
    factors.push_back(f);
    model[v] = f;
  }

  // edge potentials
  foreach (undirected_edge<finite_variable*> e, model.edges()) {
    table_factor f = gen(make_domain(e.source(), e.target()), rng);
    factors.push_back(f);
    model[e] = f;
  } 

  // run exact inference
  shafer_shenoy<table_factor> sp(factors);
  std::cout << "Tree width of the model: " << sp.tree_width() << std::endl;
  sp.calibrate();
  sp.normalize();
  std::cout << "Finished exact inference" << std::endl;

  // run mean field inference
  mean_field_pairwise<canonical_matrix<> > mf(&model);
  double diff;
  for (size_t it = 0; it < niters; ++it) {
    diff = mf.iterate();
    std::cout << "Iteration " << it << ": " << diff << std::endl;
  }
  BOOST_CHECK_LT(diff, 1e-4);
  
  // compute the KL divergence from exact to mean field
  double kl = 0.0;
  foreach (finite_variable* v, model.vertices()) {
    probability_matrix<> exact(sp.belief(make_domain(v)));
    kl += exact.relative_entropy(mf.belief(v));
  }
  kl /= model.num_vertices();
  std::cout << "Average kl = " << kl << std::endl;
  BOOST_CHECK_LT(kl, 0.02);
}
