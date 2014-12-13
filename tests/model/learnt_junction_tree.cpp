#define BOOST_TEST_MODULE learnt_junction_tree
#include <boost/test/unit_test.hpp>

#include <functional>
#include <set>

#include <boost/array.hpp>

#include <sill/graph/undirected_graph.hpp>
#include <sill/graph/algorithm/min_degree_strategy.hpp>
#include <sill/graph/algorithm/min_fill_strategy.hpp>
#include <sill/model/learnt_junction_tree.hpp>


using namespace sill;

BOOST_AUTO_TEST_CASE(test_mutation) {
  typedef undirected_graph<size_t> graph_type;

  // Build the graph.  Note that this graph must have no self-loops or
  // parallel edges.
  typedef std::pair<size_t, size_t> E;
  boost::array<E, 8> edges =
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 5), E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  graph_type g(edges);

  // Build a junction tree for the graph.
  learnt_junction_tree<size_t> jt(g, min_degree_strategy());

  BOOST_CHECK_EQUAL(jt.num_vertices(), 3);
  BOOST_CHECK_EQUAL(jt.num_edges(), 2);
  jt.check_validity();

  // Make some changes.
  std::set<size_t> c0;
  c0.insert(0);
  learnt_junction_tree<size_t>::vertex v0 = jt.find_clique_cover(c0);

  std::set<size_t> c06;
  c06.insert(0);
  c06.insert(6);
  learnt_junction_tree<size_t>::vertex v06 = jt.add_clique(c06);

  jt.add_edge(v0,v06);

  // Test the changes
  BOOST_CHECK_EQUAL(jt.num_vertices(), 4);
  BOOST_CHECK_EQUAL(jt.num_edges(), 3);
  jt.check_validity();
}
