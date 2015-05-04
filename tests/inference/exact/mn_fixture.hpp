#ifndef SILL_TEST_MN_FIXTURE_HPP
#define SILL_TEST_MN_FIXTURE_HPP

#include <sill/argument/universe.hpp>
#include <sill/factor/probability_table.hpp>
#include <sill/factor/random/diagonal_table_generator.hpp>
#include <sill/factor/random/uniform_table_generator.hpp>
#include <sill/factor/random/functional.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/graph/special/grid_graph.hpp>
#include <sill/model/pairwise_markov_network.hpp>
#include <sill/inference/exact/variable_elimination.hpp>

#include <random>

using namespace sill;

struct fixture {
  fixture() {
    size_t m = 5;
    size_t n = 4;
    vars = u.new_finite_variables(m * n, "v", 2);
    std::mt19937 rng;
    make_grid_graph(vars, m, n, mn);
    mn.initialize(marginal_fn(uniform_table_generator<ptable>(), rng),
                  marginal_fn(diagonal_table_generator<ptable>(), rng));
  }

  void check_belief(const ptable& belief, double tol) {
    std::list<ptable> factors(mn.begin(), mn.end());
    variable_elimination(factors, belief.arguments(), sum_product<ptable>());
    ptable expected = prod_all(factors).marginal(belief.arguments());
    BOOST_CHECK_SMALL(max_diff(belief, expected), tol);
  }

  void check_belief_normalized(const ptable& belief, double tol) {
    std::list<ptable> factors(mn.begin(), mn.end());
    variable_elimination(factors, belief.arguments(), sum_product<ptable>());
    ptable expected = prod_all(factors).marginal(belief.arguments());
    BOOST_CHECK_SMALL(max_diff(belief, expected.normalize()), tol);
  }

  universe u;
  domain vars;
  pairwise_markov_network<ptable> mn;
};

#endif

