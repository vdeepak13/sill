#define BOOST_TEST_MODULE junction_tree_inference
#include <boost/test/unit_test.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/graph/min_degree_strategy.hpp>
#include <sill/inference/junction_tree_inference.hpp>
#include <sill/inference/variable_elimination.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

struct fixture {
  fixture() {
    size_t m = 5;
    size_t n = 4;
    finite_var_vector variables = u.new_finite_variables(m*n, 2);
    boost::mt19937 rng;
    make_grid_graph(variables, m, n, mn);
    random_ising_model(mn, rng);
    factors.assign(mn.factors().begin(), mn.factors().end());
  }

  void check_belief(const table_factor& belief, double tol) {
    sum_product<table_factor> sp;
    min_degree_strategy strategy;
    table_factor expected = 
      variable_elimination(factors, belief.arguments(), sp, strategy);
    BOOST_CHECK_SMALL(norm_inf(belief, expected), tol);
  }

  void check_beliefs(const std::vector<table_factor>& beliefs, double tol) {
    foreach(const table_factor& belief, beliefs) {
      check_belief(belief, tol);
    }
  }

  void check_normalized(const std::vector<table_factor>& beliefs) {
    foreach(const table_factor& belief, beliefs) {
      BOOST_CHECK_CLOSE(belief.norm_constant(), 1.0, 1e-5);
    }
  }

  universe u;
  pairwise_markov_network<table_factor> mn;
  std::vector<table_factor> factors;
};

BOOST_FIXTURE_TEST_CASE(test_shafer_shenoy, fixture) {
  shafer_shenoy<table_factor> mn_engine(mn);
  mn_engine.calibrate();
  check_beliefs(mn_engine.clique_beliefs(), 1e-10);
  
  shafer_shenoy<table_factor> fac_engine(factors);
  fac_engine.calibrate();
  check_beliefs(fac_engine.clique_beliefs(), 1e-10);

  fac_engine.normalize();
  check_normalized(fac_engine.clique_beliefs());

  foreach(undirected_edge<finite_variable*> e, mn.edges()) {
    fac_engine.belief(mn.nodes(e));
  }

  // TODO: conditioning
}

BOOST_FIXTURE_TEST_CASE(test_hugin, fixture) {
  hugin<table_factor> mn_engine(mn);
  mn_engine.calibrate();
  check_beliefs(mn_engine.clique_beliefs(), 1e-10);
  
  hugin<table_factor> fac_engine(factors);
  fac_engine.calibrate();
  check_beliefs(fac_engine.clique_beliefs(), 1e-10);

  fac_engine.normalize();
  check_normalized(fac_engine.clique_beliefs());
}

