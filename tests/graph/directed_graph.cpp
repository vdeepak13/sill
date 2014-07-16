#define BOOST_TEST_MODULE directed_graph
#include <boost/test/unit_test.hpp>

#include <sill/graph/directed_graph.hpp>
#include <sill/graph/graph_traversal.hpp>

#include <vector>
#include <algorithm>
#include <set>

#include <boost/array.hpp>
#include <boost/mpl/list.hpp>

#include "predicates.hpp"

#include <sill/macros_def.hpp>

using namespace sill;
using std::make_pair;

typedef size_t V;
typedef size_t vertex_property;
typedef size_t edge_property;
typedef directed_graph<V, vertex_property, edge_property> graph;
typedef std::pair<V,V> E;

template class directed_graph<size_t>;
template class directed_graph<size_t, double, double>;

BOOST_TEST_DONT_PRINT_LOG_VALUE(::graph::edge_iterator);
BOOST_TEST_DONT_PRINT_LOG_VALUE(::graph::vertex_iterator);
// see http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/utf/user-guide/test-output/test-log.html

BOOST_AUTO_TEST_CASE(test_directed_edge) {
  directed_edge<V> e1, e2;
  BOOST_CHECK(e1 == e2);
  BOOST_CHECK(e1.source() == e2.source());
  BOOST_CHECK(e1.source() == e1.target());
}


BOOST_AUTO_TEST_CASE(test_constructors) {
  // default constructor
  graph g1;
  BOOST_CHECK(g1.empty());
  BOOST_CHECK_EQUAL(g1.edges().first, g1.edges().second);
  BOOST_CHECK_EQUAL(g1.vertices().first, g1.vertices().second);

  // edge list constructor
  boost::array<E, 8> edges = 
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  graph g2(edges);
  foreach(E e, edges) {
    BOOST_CHECK(g2.contains(e.first));
    BOOST_CHECK(g2.contains(e.second));
    BOOST_CHECK(g2.contains(e.first, e.second));
    BOOST_CHECK(!g2.contains(e.second, e.first));
  }
  BOOST_CHECK(!g2.contains(8,2));
  BOOST_CHECK(!g2.contains(8));

  // copy constructor
  graph g3(g2);
  foreach(E e, edges) {
    BOOST_CHECK(g3.contains(e.first));
    BOOST_CHECK(g3.contains(e.second));
    BOOST_CHECK(g3.contains(e.first, e.second));
    BOOST_CHECK(!g3.contains(e.second, e.first));
  }
  BOOST_CHECK(!g3.contains(8,2));
  BOOST_CHECK(!g3.contains(8));
}


BOOST_AUTO_TEST_CASE(test_vertices) {
  graph g;
  boost::array<V, 11> verts = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
  std::map<V,V> vert_map;
  foreach(V v, verts) {
    vert_map[v] = v;
    g.add_vertex(v,v);
  }
  foreach(V v, g.vertices()) {
    BOOST_CHECK(vert_map.count(v));
    BOOST_CHECK_EQUAL(g[v], vert_map[v]);
    vert_map.erase(v);
  }
  BOOST_CHECK(vert_map.empty());
}


BOOST_AUTO_TEST_CASE(test_edges) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(1, 7), E(5, 8)}};
  std::map<E,size_t> data;
  graph g;
  size_t i = 0; 
  foreach(E e, edges) {
    g.add_edge(e.first, e.second, i);
    data[e] = i; 
    ++i;
  }
  foreach(graph::edge e, g.edges()) {
    BOOST_CHECK(data.count(make_pair(e.source(), e.target())));
    BOOST_CHECK_EQUAL(data[make_pair(e.source(), e.target())], g[e]);
    data.erase(make_pair(e.source(), e.target()));
  }
  BOOST_CHECK(data.empty());
}


BOOST_AUTO_TEST_CASE(test_parents) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(4, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(1, 4),
      E(2, 6), E(3, 5), E(7, 4), E(5, 8)}};
  graph g(edges);
  boost::array<V, 3> verts = {{1,3,7}};
  std::set<V> vert_set;
  vert_set.insert(verts.begin(), verts.end());
  foreach(V v, g.parents(4)) {
    BOOST_CHECK(vert_set.count(v));
    vert_set.erase(v);
  }
  BOOST_CHECK(vert_set.empty());
}


BOOST_AUTO_TEST_CASE(test_children) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(4, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(4, 7), E(5, 8)}};
  graph g(edges);
  boost::array<V, 4> verts = {{0,1,7,9}};
  std::set<V> vert_set;
  vert_set.insert(verts.begin(), verts.end());
  foreach(V v, g.children(4)) {
    BOOST_CHECK(vert_set.count(v));
    vert_set.erase(v);
  }
  BOOST_CHECK(vert_set.empty());
}


BOOST_AUTO_TEST_CASE(test_in_edges) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g;
  std::map<E,size_t> edge_values;
  V u = 3;
  size_t i = 0;
  foreach(E e, edges) { 
    if(e.second == u ) {
      g.add_edge(e.first, e.second, i);
      edge_values[e] = i;
      ++i;
    }
  }
  foreach(graph::edge edge, g.in_edges(3)) { 
    E e = make_pair(edge.source(), edge.target());
    BOOST_CHECK(edge_values.count(e));
    BOOST_CHECK_EQUAL(g[edge], edge_values[e]);
    edge_values.erase(e);
  }
  BOOST_CHECK(edge_values.empty());
}


BOOST_AUTO_TEST_CASE(test_out_edges) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(6, 3), E(3, 5), E(7, 3), E(3, 8)}};
  graph g;
  std::map<E,size_t> edge_values;
  V u = 3;
  size_t i = 0;
  foreach(E e, edges) { 
    if(e.first == u ) {
      g.add_edge(e.first, e.second, i);
      edge_values[e] = i;
      ++i;
    }
  }
  foreach(graph::edge edge, g.out_edges(3)) { 
    E e = make_pair(edge.source(), edge.target());
    BOOST_CHECK(edge_values.count(e));
    BOOST_CHECK_EQUAL(g[edge], edge_values[e]);
    edge_values.erase(e);
  }
  BOOST_CHECK(edge_values.empty());
}


BOOST_AUTO_TEST_CASE(test_contains) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g(edges);
  for (size_t i = 0; i <= 10; ++i) {
    BOOST_CHECK(g.contains(i));
  }
  BOOST_CHECK(!g.contains(11));
  foreach(E e, edges) {
    BOOST_CHECK(g.contains(e.first, e.second));
    BOOST_CHECK(!g.contains(e.second, e.first));
    BOOST_CHECK(g.contains(g.get_edge(e.first, e.second)));
  }
  BOOST_CHECK(!g.contains(0, 3));
  BOOST_CHECK(!g.contains(2, 10));
}


BOOST_AUTO_TEST_CASE(test_get_edge) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(1, 7), E(5, 8)}};
  std::map<E,size_t> data;
  graph g;
  size_t i = 0; 
  foreach(E e, edges) {
    g.add_edge(e.first, e.second, i);
    data[e] = i; 
    ++i;
  }
  foreach(E e, edges) {
    BOOST_CHECK_EQUAL(g.get_edge(e.first, e.second).source(), e.first);
    BOOST_CHECK_EQUAL(g.get_edge(e.first, e.second).target(), e.second);
    BOOST_CHECK_EQUAL(g[g.get_edge(e.first, e.second)], data[e]);
  }
}


BOOST_AUTO_TEST_CASE(test_degree) {
  graph g;
  g.add_vertex(2,3);
  BOOST_CHECK_EQUAL(g.in_degree(2), 0);
  BOOST_CHECK_EQUAL(g.out_degree(2), 0);
  BOOST_CHECK_EQUAL(g.degree(2), 0);

  g.add_edge(1,2,3);
  BOOST_CHECK_EQUAL(g.in_degree(2), 1);
  BOOST_CHECK_EQUAL(g.in_degree(1), 0);
  BOOST_CHECK_EQUAL(g.out_degree(1), 1);
  BOOST_CHECK_EQUAL(g.out_degree(2), 0);
  BOOST_CHECK_EQUAL(g.degree(1), 1);
  BOOST_CHECK_EQUAL(g.degree(2), 1);

  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g2(edges);
  BOOST_CHECK_EQUAL(g2.in_degree(3), 3);
  BOOST_CHECK_EQUAL(g2.out_degree(4), 2);
  BOOST_CHECK_EQUAL(g2.in_degree(3) + g2.out_degree(3), g2.degree(3));
}


BOOST_AUTO_TEST_CASE(test_num) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g(edges);
  BOOST_CHECK_EQUAL(g.num_vertices(), 11);
  BOOST_CHECK_EQUAL(g.num_edges(), 12);
  g.clear_in_edges(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 11);
  BOOST_CHECK_EQUAL(g.num_edges(), 9);
  g.clear_out_edges(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 11);
  BOOST_CHECK_EQUAL(g.num_edges(), 6);

  g.remove_vertex(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 10);
  BOOST_CHECK_EQUAL(g.num_edges(), 6);
  g.remove_vertex(0);
  BOOST_CHECK_EQUAL(g.num_vertices(), 9);
  BOOST_CHECK_EQUAL(g.num_edges(), 4);
}


typedef boost::mpl::list<int, double, void_> test_types;
BOOST_AUTO_TEST_CASE_TEMPLATE(test_comparison, EP, test_types) {
  typedef directed_graph<size_t, int, EP> Graph;

  Graph g1;
  g1.add_vertex(1, 0);
  g1.add_vertex(2, 1);
  g1.add_vertex(3, 2);
  g1.add_edge(1, 2);
  g1.add_edge(2, 3);
  
  Graph g2;
  g2.add_vertex(1, 0);
  g2.add_vertex(3, 2);
  g2.add_vertex(2, 1);
  g2.add_edge(1, 2);
  g2.add_edge(2, 3);

  BOOST_CHECK_EQUAL(g1, g2);

  Graph g3 = g2;
  g3[1] = -1;
  BOOST_CHECK_NE(g2, g3);

  Graph g4 = g2;
  g4.remove_edge(2, 3);
  g4.add_edge(1, 3);
  BOOST_CHECK_NE(g2, g3);
}


BOOST_AUTO_TEST_CASE(test_serialization) {
  directed_graph<int, std::string, double> g;
  g.add_vertex(1, "hello");
  g.add_vertex(2, "bye");
  g.add_vertex(3, "maybe");
  g.add_edge(1, 2, 1.5);
  g.add_edge(2, 3, 2.5);
  g.add_edge(3, 2, 3.5);
  BOOST_CHECK(serialize_deserialize(g));
}


// the following test is too slow for a unit test;
// it should be in timings instead
// BOOST_AUTO_TEST_CASE(test_large) {
//   size_t n = 100000;
//   size_t m = 100;  
//   directed_graph<size_t, size_t, size_t> g;
//   for(size_t u = 0; u < n; u++) {
//     g.add_vertex(u,u);
//     for(size_t v = 0; v < m; v++) {
//       g.add_vertex(v,v);
//       g.add_edge(u,v, u * v);
//     } 
//   }
// }
