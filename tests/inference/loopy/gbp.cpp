#define BOOST_TEST_MODULE gbp
#include <boost/test/unit_test.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/random/functional.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/special/grid_graph.hpp>

#include <sill/inference/loopy/asynchronous_gbp.hpp>
#include <sill/inference/loopy/asynchronous_gbp_pc.hpp>
#include <sill/inference/loopy/kikuchi.hpp>
#include <sill/inference/loopy/bethe.hpp>

#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>
#include <sill/model/decomposable.hpp>

#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {
  template class asynchronous_gbp<table_factor>;
  template class asynchronous_gbp_pc<table_factor>;
  
  template class asynchronous_gbp<canonical_gaussian>;
  template class asynchronous_gbp_pc<canonical_gaussian>;
}

using namespace sill;

template <typename Engine>
void test(const pairwise_markov_network<table_factor>& model,
          const region_graph<finite_variable*, table_factor>& rg,
          size_t niters,
          const decomposable<table_factor>& joint,
          double error_tol,
          double diff_tol) {
  // run the inference
  Engine engine(rg);
  engine.initialize_factors(model);
  double residual = 0.0;
  for(size_t i = 0; i < niters; i++) {
    residual = engine.iterate(0.5);
  }
  BOOST_CHECK_SMALL(residual, 0.1);
  
  // iterate
  for (size_t it = 0; it < niters; ++it) {
    engine.iterate(0.5);
  }
  
  // check if the approximation error is small enough
  double max_error = 0;
  foreach(finite_variable* var, joint.arguments()) {
    table_factor exact = joint.marginal(make_domain(var));
    table_factor approx = engine.belief(make_domain(var));
    double error = norm_inf(exact, approx);
    max_error = std::max(max_error, error);
    BOOST_CHECK_SMALL(error, error_tol);
  }
  std::cout << "Maximum error: " << max_error << std::endl;

  // Check if the edge marginals agree
  double max_diff = 0;
  foreach(directed_edge<size_t> e, rg.edges()) {
    table_factor sbel =
      engine.belief(e.source()).marginal(rg.cluster(e.target()));
    table_factor tbel =
      engine.belief(e.target());
    double diff = norm_inf(sbel, tbel);
    max_diff = std::max(max_diff, diff);
    BOOST_CHECK_SMALL(diff, diff_tol);
  }
  std::cout << "Belief consistency: " << max_diff << std::endl;
}

struct fixture {
  fixture() {
    size_t m = 5;
    size_t n = 4;

    // generate a random model
    finite_var_vector varvec = u.new_finite_variables(m*n, 2);
    arma::field<finite_variable*> vars = make_grid_graph(varvec, m, n, mn);
    uniform_factor_generator gen;
    boost::mt19937 rng;
    mn.initialize(marginal_fn(gen, rng));

    // create a region graph with clusters over pairs of adjacent variables
    std::vector<finite_domain> clusters;
    foreach(const table_factor& f, mn.factors()) {
      clusters.push_back(f.arguments());
    }
    bethe(clusters, pairs_rg);

    // create a region graph with clusters over 2x2 adjacent variables
    std::vector<finite_domain> root_clusters;
    for(size_t i = 0; i < m-1; i++)
      for(size_t j = 0; j < n-1; j++) {
        finite_domain cluster = 
          make_domain(vars(i,j), vars(i+1,j), vars(i,j+1), vars(i+1,j+1));
        root_clusters.push_back(cluster);
      }
    kikuchi(root_clusters, square_rg);

    // compute the joint distribution
    dm *= mn.factors();
  }

  universe u;
  pairwise_markov_network<table_factor> mn;
  region_graph<finite_variable*, table_factor> pairs_rg;
  region_graph<finite_variable*, table_factor> square_rg;
  decomposable<table_factor> dm;
};

BOOST_FIXTURE_TEST_CASE(test_parent_to_child, fixture) {
  test<asynchronous_gbp_pc<table_factor> >(mn, pairs_rg, 30, dm, 1e-1, 1e-8);
  test<asynchronous_gbp_pc<table_factor> >(mn, square_rg, 30, dm, 1e-3, 1e-3);
}

BOOST_FIXTURE_TEST_CASE(test_two_way, fixture) {
  test<asynchronous_gbp<table_factor> >(mn, pairs_rg, 30, dm, 1e-1, 1e-6);
  test<asynchronous_gbp<table_factor> >(mn, square_rg, 30, dm, 1e-3, 1e-3);
}
