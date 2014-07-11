#define BOOST_TEST_MODULE junction_tree
#include <boost/test/unit_test.hpp>

#include <set>

#include <boost/array.hpp>

#include <sill/graph/undirected_graph.hpp>
#include <sill/graph/min_degree_strategy.hpp>
#include <sill/model/junction_tree.hpp>

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::set<size_t>);

BOOST_AUTO_TEST_CASE(test_operations) {  
  using namespace sill;

  // Build the graph.  Note that this graph must have no self-loops or
  // parallel edges.
  typedef std::pair<size_t, size_t> E;
  boost::array<E, 8> edges =
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 5), E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  undirected_graph<size_t> g(edges);

  typedef std::set<size_t> node_set;

  // Build a junction tree using the min-degree strategy
  junction_tree<size_t> jt(g, min_degree_strategy());
  jt.check_validity();

  boost::array<size_t, 2> clique1 = {1, 5};
  boost::array<size_t, 3> clique2 = {0, 2, 4};
  boost::array<size_t, 4> clique3 = {1, 2, 3, 4};
  BOOST_CHECK_EQUAL(jt.num_vertices(), 3);
  BOOST_CHECK_EQUAL(jt.clique(1), node_set(clique1.begin(), clique1.end()));
  BOOST_CHECK_EQUAL(jt.clique(2), node_set(clique2.begin(), clique2.end()));
  BOOST_CHECK_EQUAL(jt.clique(3), node_set(clique3.begin(), clique3.end()));
  BOOST_CHECK(jt.contains(1, 3));
  BOOST_CHECK(jt.contains(2, 3));

  // Copy the junction tree and check validity of original and copy.
  junction_tree<size_t> jt2(jt);
  jt.check_validity();
  jt2.check_validity();
  BOOST_CHECK_EQUAL(jt, jt2);

  // Check the subtree cover for the set {0, 5}
  node_set nodes05; nodes05.insert(5); nodes05.insert(0);
  jt.mark_subtree_cover(nodes05, true);
  BOOST_CHECK(jt.marked(1));
  BOOST_CHECK(jt.marked(2));
  BOOST_CHECK(jt.marked(3));
  BOOST_CHECK(jt.marked(jt.get_edge(1, 3)));
  BOOST_CHECK(jt.marked(jt.get_edge(2, 3)));

  // Check the subtree cover for the set {1}
  node_set nodes1; nodes1.insert(1);
  jt.mark_subtree_cover(nodes1, true);
  BOOST_CHECK(jt.marked(1) != jt.marked(3));
  BOOST_CHECK(!jt.marked(2));
  BOOST_CHECK(!jt.marked(jt.get_edge(1, 3)));
  BOOST_CHECK(!jt.marked(jt.get_edge(2, 3)));
}
