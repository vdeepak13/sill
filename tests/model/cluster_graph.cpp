#define BOOST_TEST_MODULE cluster_graph
#include <boost/test/unit_test.hpp>

#include <functional>

#include <sill/model/cluster_graph.hpp>
#include <sill/base/universe.hpp>

#include "predicates.hpp"

using namespace sill;

struct fixture {
  fixture()
    : v(u.new_finite_variables(6, 2)) {
    cg.add_cluster(1, make_domain(v[0], v[1]));
    cg.add_cluster(2, make_domain(v[1], v[2], v[3]));
    cg.add_cluster(3, make_domain(v[2], v[3], v[4]));
    cg.add_cluster(4, make_domain(v[3], v[5]));
    cg.add_edge(1, 2);
    cg.add_edge(2, 3);
    cg.add_edge(2, 4);
  } 

  universe u;
  finite_var_vector v;
  cluster_graph<finite_variable*> cg;
};

BOOST_FIXTURE_TEST_CASE(test_properties, fixture) {
  BOOST_CHECK(cg.connected());
  BOOST_CHECK(cg.running_intersection());

  cg.remove_edge(2, 4);
  BOOST_CHECK(!cg.connected());
  BOOST_CHECK(!cg.running_intersection());
}

BOOST_FIXTURE_TEST_CASE(test_subgraph, fixture) {
  cluster_graph<finite_variable*> subgraph;
  subgraph = cg.subgraph(1, 5);
  BOOST_CHECK_EQUAL(cg, subgraph);

  subgraph = cg.subgraph(1, 0);
  cluster_graph<finite_variable*> subgraph1;
  subgraph1.add_cluster(1, make_domain(v[0], v[1]));
  BOOST_CHECK_EQUAL(subgraph1, subgraph);
}

BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(cg, u));
}
