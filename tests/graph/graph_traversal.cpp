#define BOOST_TEST_MODULE graph_traversal
#include <boost/test/unit_test.hpp>

#include <sill/graph/directed_graph.hpp>
#include <sill/graph/directed_multigraph.hpp>
#include <sill/graph/graph_traversal.hpp>

#include <boost/array.hpp>
#include <boost/mpl/list.hpp>

#include "predicates.hpp"

using namespace sill;

typedef boost::mpl::list<
  directed_graph<int, int, double>,
  directed_multigraph<int, int, double>
> graph_types;


BOOST_AUTO_TEST_CASE_TEMPLATE(test_simple, Graph, graph_types) {
  typedef typename Graph::vertex V;
  typedef std::pair<V, V> E;
  boost::array<E, 6> edges = 
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), 
      E(2, 3), E(3, 4)}};
  Graph g(edges);
  std::vector<V> order = directed_partial_vertex_order(g);
  BOOST_CHECK(is_partial_vertex_order(order, g));
}


BOOST_AUTO_TEST_CASE_TEMPLATE(test_multi_edges, Graph, graph_types) {
  typedef typename Graph::vertex V;
  typedef std::pair<V, V> E;
  boost::array<E, 7> edges = 
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), 
      E(2, 3), E(3, 4), E(1, 2)}};
  Graph g(edges);
  std::vector<V> order = directed_partial_vertex_order(g);
  BOOST_CHECK(is_partial_vertex_order(order, g));
}


// TODO: change the API of directed_partial_vertex_order,
//       so that it returns an error or throws exception on failure

// BOOST_AUTO_TEST_CASE_TEMPLATE(test_cycle, Graph, graph_types) {
//   typedef typename Graph::vertex V;
//   typedef std::pair<V, V> E;
//   boost::array<E, 7> edges = 
//     {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), 
//       E(2, 3), E(3, 4), E(4, 1)}};
//   Graph g(edges);
//   std::vector<V> order = directed_partial_vertex_order(g);
//   BOOST_CHECK(is_partial_vertex_order(order, g));
// }
