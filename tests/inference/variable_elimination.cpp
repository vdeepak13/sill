#define BOOST_TEST_MODULE variable_elimination
#include <boost/test/unit_test.hpp>

#include <sill/factor/random/functional.hpp>
#include <sill/factor/random/ising_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/graph/min_degree_strategy.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>

#include "../factor/predicates.hpp"

#include <sill/macros_def.hpp>

BOOST_AUTO_TEST_CASE(test_grid) {
  using namespace sill;

  size_t m = 4;
  size_t n = 3;

  universe u;                      
  finite_var_vector variables = u.new_finite_variables(m*n, 2);

  boost::mt19937 rng;
  ising_factor_generator gen(0.0, 0.5, 0.0, 1.0);
  pairwise_markov_network<table_factor> mn;
  make_grid_graph(variables, m, n, mn);
  mn.initialize(marginal_fn(gen, rng));

  std::vector<table_factor> factors(mn.factors().begin(), mn.factors().end());
  table_factor product = prod_all(factors);

  typedef undirected_edge<finite_variable*> edge_type;
  foreach (edge_type e, mn.edges()) {
    sum_product<table_factor> sp;
    finite_domain retain = mn.nodes(e);
    table_factor elim_result = 
      variable_elimination(factors, retain, sp, min_degree_strategy());
    table_factor direct_result = product.marginal(retain);
    BOOST_CHECK(are_close(elim_result, direct_result, 1e-3));
  }
}
