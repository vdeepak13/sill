#define BOOST_TEST_MODULE undirected_graph
#include <boost/test/unit_test.hpp>

#include <set>
#include <map>
#include <boost/array.hpp> 

#include <sill/graph/undirected_graph.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
using std::make_pair;

typedef size_t V;
typedef undirected_graph<V, size_t, size_t> graph;
typedef std::pair<V, V> E;

BOOST_TEST_DONT_PRINT_LOG_VALUE(::graph::edge_iterator);
BOOST_TEST_DONT_PRINT_LOG_VALUE(::graph::vertex_iterator);
// see http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/utf/user-guide/test-output/test-log.html

BOOST_AUTO_TEST_CASE(test_undirected_edge) {
  undirected_edge<V> e1, e2;
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
  // iterators cannot be printed, so we do not use BOOST_CHECK_EQUAL here
  // for an alternative
  
  // edge list constructor
  boost::array<E, 8> edges = 
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  graph g2(edges);
  foreach(E e, edges) {
    BOOST_CHECK(g2.contains(e.first));
    BOOST_CHECK(g2.contains(e.second));
    BOOST_CHECK(g2.contains(e.first, e.second));
    BOOST_CHECK(g2.contains(e.second, e.first));
  }
  BOOST_CHECK(!g2.contains(8, 2));
  BOOST_CHECK(!g2.contains(8));

  //copy constructor
  graph g3(g2);
  foreach(E e, edges) {
    BOOST_CHECK(g3.contains(e.first));
    BOOST_CHECK(g3.contains(e.second));
    BOOST_CHECK(g3.contains(e.first, e.second));
    BOOST_CHECK(g3.contains(e.second, e.first));
  }
  BOOST_CHECK(!g3.contains(8,2));
  BOOST_CHECK(!g3.contains(8));
}


BOOST_AUTO_TEST_CASE(test_vertices) {
  graph g;
  boost::array<V, 11> verts = {{0,1,2,3,4,5,6,7,8,9,10}};
  std::map<V,V> vert_map;
  foreach(V v, verts) {
    vert_map[v] = v;
    g.add_vertex(v,v);
  }
  foreach(V v, g.vertices()) {
    BOOST_CHECK(vert_map.count(v) == 1);
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
  foreach(graph::edge edge, g.edges()) {
    E e = make_pair(edge.source(), edge.target());
    E erev = make_pair(edge.target(), edge.source());
    BOOST_CHECK((data.count(e) == 1) ^ (data.count(erev) == 1));
    if(data.count(erev) == 1) e = erev;
    BOOST_CHECK_EQUAL(data[e], g[edge]);
    data.erase(e);
  }
  BOOST_CHECK(data.empty());
}


BOOST_AUTO_TEST_CASE(test_neighbors) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(4, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(4, 7), E(5, 8)}};
  graph g(edges);
  boost::array<V, 5> verts = {{0,1,3,7,9}};
  std::set<V> neighbors;
  neighbors.insert(verts.begin(), verts.end());
  foreach(V v, g.neighbors(4))  {
    BOOST_CHECK(neighbors.count(v));
    neighbors.erase(v);
  }
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
    if(e.first == u || e.second == u) {
      g.add_edge(e.first, e.second, i);
      if(e.first == u) e = make_pair(e.second, e.first);
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
    if(e.first == u  || e.second == u) {
      g.add_edge(e.first, e.second, i);
      if(e.second == u) e = make_pair(e.second, e.first);
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
    BOOST_CHECK(g.contains(e.second, e.first));
    BOOST_CHECK(g.contains(g.get_edge(e.first, e.second)));
    BOOST_CHECK(g.contains(g.get_edge(e.second, e.first)));
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
    data[make_pair(e.second, e.first)] = i; 
    ++i;
  }
  foreach(E e, edges) {
    BOOST_CHECK_EQUAL(g.get_edge(e.first, e.second).source(), e.first);
    BOOST_CHECK_EQUAL(g.get_edge(e.first, e.second).target(), e.second);
    BOOST_CHECK_EQUAL(g[g.get_edge(e.first, e.second)], data[e]);
  }
}


BOOST_AUTO_TEST_CASE(test_degree) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g(edges);
  BOOST_CHECK_EQUAL(g.in_degree(3), g.out_degree(3));
  BOOST_CHECK_EQUAL(g.out_degree(3), g.degree(3));
  BOOST_CHECK_EQUAL(g.degree(3), 6);
}


BOOST_AUTO_TEST_CASE(test_num) {
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g(edges);
  BOOST_CHECK_EQUAL(g.num_vertices(), 11);
  BOOST_CHECK_EQUAL(g.num_edges(), 12);
  g.clear_edges(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 11);
  BOOST_CHECK_EQUAL(g.num_edges(), 6);
  g.remove_vertex(3);
  BOOST_CHECK_EQUAL(g.num_vertices(), 10);
  BOOST_CHECK_EQUAL(g.num_edges(), 6);
  g.remove_vertex(0);
  BOOST_CHECK_EQUAL(g.num_vertices(), 9);
  BOOST_CHECK_EQUAL(g.num_edges(), 4);
}
