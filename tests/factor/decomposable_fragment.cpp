#include <boost/test/unit_test.hpp>

#include <boost/bind.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/decomposable_fragment.hpp>
#include <sill/factor/random/functional.hpp>
#include <sill/factor/random/ising_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/inference/junction_tree_inference.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>

#include "predicates.hpp"

using namespace boost::unit_test;
using namespace sill;

template class decomposable_fragment<table_factor>;
template class decomposable_fragment<canonical_gaussian>;

void test_marginal(size_t m, size_t n) {
  assert(m * m >= 4); // we need at least 4 variables

  boost::mt19937 rng;
  universe u;

  // generate a random model
  pairwise_markov_network< table_factor > mn;
  finite_var_vector variables = u.new_finite_variables(m * n, 2);
  make_grid_graph(variables, m, n, mn);
  mn.initialize(marginal_fn(ising_factor_generator(), rng));

  // compute the corresponding decomposable fragment
  shafer_shenoy<table_factor> ss(mn);
  ss.calibrate();
  ss.normalize();
  decomposable_fragment<table_factor> df(ss.clique_beliefs());

  // compute the corresponding decomposable model
  decomposable<table_factor> dm;
  dm *= mn.factors();

  // compute the marginal over v0 and v3
  finite_domain v03 = make_domain(variables[0], variables[3]);
  decomposable_fragment<table_factor> df03 = df.marginal(v03);
  table_factor tf03 = df03.flatten();
  
  BOOST_CHECK(are_close(tf03, dm.marginal(v03), 1e-5));
}

test_suite* init_unit_test_suite(int arc, char** argv) {
  framework::master_test_suite().
    add(BOOST_TEST_CASE(boost::bind(&test_marginal, 2, 2)));
  framework::master_test_suite().
    add(BOOST_TEST_CASE(boost::bind(&test_marginal, 3, 2)));
  framework::master_test_suite().
    add(BOOST_TEST_CASE(boost::bind(&test_marginal, 2, 3)));
  framework::master_test_suite().
    add(BOOST_TEST_CASE(boost::bind(&test_marginal, 3, 3)));
  framework::master_test_suite().
    add(BOOST_TEST_CASE(boost::bind(&test_marginal, 4, 4)));
  framework::master_test_suite().
    add(BOOST_TEST_CASE(boost::bind(&test_marginal, 5, 5)));
  return 0;
}
