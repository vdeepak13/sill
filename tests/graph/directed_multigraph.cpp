// Warning: this test appears to be mostly a copy of directed_graph.cpp
// we need to add edge multiplicity in most of the cases

#define BOOST_TEST_MODULE 
#include <boost/test/unit_test.hpp>

#include <sill/graph/directed_multigraph.hpp>
#include <sill/graph/graph_traversal.hpp>

#include <algorithm>
#include <set>
#include <vector>

#include <boost/array.hpp>

#include "predicates.hpp"

#include <sill/macros_def.hpp>

using namespace sill;
using namespace std;

typedef size_t V;
typedef size_t vertex_property;
typedef size_t edge_property;
typedef directed_multigraph<V, vertex_property, edge_property> graph;
typedef pair<V,V> E;

template class directed_multigraph<size_t>;
template class directed_multigraph<size_t, double, double>;

BOOST_TEST_DONT_PRINT_LOG_VALUE(::graph::edge_iterator);
BOOST_TEST_DONT_PRINT_LOG_VALUE(::graph::vertex_iterator);
// see http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/utf/user-guide/test-output/test-log.html


BOOST_AUTO_TEST_CASE(test_directed_edge) {
  directed_edge<V> e1, e2;
  BOOST_CHECK_EQUAL(e1, e2);
  BOOST_CHECK_EQUAL(e1.source(), e2.source());
  BOOST_CHECK_EQUAL(e1.source(), e1.target());
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
  BOOST_CHECK(!g2.contains(8, 2));
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
    BOOST_CHECK_EQUAL(vert_map.count(v), 1);
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
    BOOST_CHECK_EQUAL(data.count(make_pair(e.source(), e.target())), 1);
    BOOST_CHECK_EQUAL(data[make_pair(e.source(), e.target())], g[e]);
    data.erase(make_pair(e.source(), e.target()));
  }
  BOOST_CHECK(data.empty());

  // test edge removal
  g.remove_edge(1, 9);
  g.remove_edge(1, 3);
  i = 0;
  foreach(E e, edges) {
    if (e != E(1, 9) && e != E(1,3))
      data[e] = i;
    ++i;
  }  
  foreach(graph::edge e, g.edges()) {
    BOOST_CHECK_EQUAL(data.count(make_pair(e.source(), e.target())), 1);
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
    BOOST_CHECK_EQUAL(vert_set.count(v), 1);
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
    BOOST_CHECK_EQUAL(vert_set.count(v), 1);
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
    BOOST_CHECK_EQUAL(edge_values.count(e), 1);
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
    BOOST_CHECK_EQUAL(edge_values.count(e), 1);
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


BOOST_AUTO_TEST_CASE(test_comparison) {
  typedef directed_multigraph<size_t, int, double> Graph;

  Graph g1;
  g1.add_vertex(1, 0);
  g1.add_vertex(2, 1);
  g1.add_vertex(3, 2);
  g1.add_edge(1, 2);
  g1.add_edge(2, 3);
  g1.add_edge(2, 3, 5.0);
  
  Graph g2;
  g2.add_vertex(1, 0);
  g2.add_vertex(3, 2);
  g2.add_vertex(2, 1);
  g2.add_edge(1, 2);
  g2.add_edge(2, 3, 5.0);
  g2.add_edge(2, 3);

  BOOST_CHECK_EQUAL(g1, g2);

  Graph g3 = g2;
  g3[1] = -1;
  BOOST_CHECK_NE(g2, g3);

  Graph g4 = g2;
  g4.remove_edge(2, 3);
  BOOST_CHECK_NE(g2, g3);
}


BOOST_AUTO_TEST_CASE(test_serialization) {
  directed_multigraph<int, std::string, double> g;
  g.add_vertex(1, "hello");
  g.add_vertex(2, "bye");
  g.add_vertex(3, "maybe");
  g.add_edge(1, 2, 1.5);
  g.add_edge(2, 3, 2.5);
  g.add_edge(3, 2, 3.5);
  g.add_edge(1, 2, 0.5);
  BOOST_CHECK(serialize_deserialize(g));
}


// This test is too slow for a unit test and does not
// actually test anything
// BOOST_AUTO_TEST_CASE(test_large) {
//   size_t n = 100000;
//   size_t m = 100;
//   cout << "Beginning Large Test" << endl;
//   directed_multigraph<size_t, size_t, size_t> g;
//   for(size_t u = 0; u < n; u++) {
//     g.add_vertex(u,u);
//     for(size_t v = 0; v < m; v++) {
//       g.add_vertex(v,v);
//       g.add_edge(u,v, u * v);
//     } 
//   }
// }
