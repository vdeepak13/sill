#define BOOST_TEST_MODULE bayes_markov_graph
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/model/bayesian_graph.hpp>
#include <sill/model/markov_graph.hpp>

#include "predicates.hpp"

using namespace sill;

template class bayesian_graph<finite_variable*>;
template class markov_graph<finite_variable*>;

template class bayesian_graph<vector_variable*>;
template class markov_graph<vector_variable*>;

struct fixture {

  fixture()
    : v(u.new_finite_variables(5, 2)),
      vars(v.begin(), v.end()),
      bg(vars),
      mg(vars) {
    /* Create graph:
     * 0 --> 4
     * 1 --> 2 3
     * 2 --> 3
     * 3 --> 4
     * 4 --> 
     */
    bg.add_edge(v[0], v[4]);
    bg.add_edge(v[1], v[2]);
    bg.add_edge(v[1], v[3]);
    bg.add_edge(v[2], v[3]);
    bg.add_edge(v[3], v[4]);

    mg.add_clique(make_domain(v[0], v[3], v[4]));
    mg.add_clique(make_domain(v[1], v[2], v[3]));
  }

  universe u;
  finite_var_vector v;
  finite_domain vars;
  bayesian_graph<finite_variable*> bg;
  markov_graph<finite_variable*> mg;

};

BOOST_FIXTURE_TEST_CASE(test_conversion, fixture) {
  BOOST_CHECK_EQUAL(mg, bayes2markov_graph(bg));
}

BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(bg, u));
  BOOST_CHECK(serialize_deserialize(mg, u));
}
