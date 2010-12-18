
#include <iostream>
#include <sill/macros_def.hpp>
#include <sill/graph/directed_graph.hpp>
#include <sill/graph/graph_traversal.hpp>
#include <vector>
#include <algorithm>
#include <set>

#include <boost/array.hpp>
#include <sill/macros_def.hpp>

using namespace sill;
using namespace std;

typedef size_t V;
typedef size_t vertex_property;
typedef size_t edge_property;
typedef directed_graph<V, vertex_property, edge_property> graph;
typedef pair<V,V> E;


void test_directed_edge() {
  cout << "Testing Directed Edge: " << endl;
  directed_edge<V> e1, e2;
  assert(e1 == e2);
  assert(e1.source() == e2.source());
  assert(e1.source() == e1.target());
}


void test_constructors(){
  cout << "Testng Constructors:" << endl;
  cout << "Testing Default constructors:" << endl;
  graph g1;
  assert(g1.empty());
  assert(g1.edges().first == g1.edges().second);
  assert(g1.vertices().first == g1.vertices().second);
  
  cout << "Testing edge list constructor: " << endl;
  boost::array<E, 8> edges = 
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  graph g2(edges);
  foreach(E e, edges) {
    assert(g2.contains(e.first));
    assert(g2.contains(e.second));
    assert(g2.contains(e.first, e.second));
    assert( !g2.contains(e.second, e.first) );
  }
  assert( !g2.contains(8,2) );
  assert( !g2.contains(8) );

  cout << "Testing copy constructor: " << endl;
  graph g3 = g2;
  foreach(E e, edges) {
    assert(g3.contains(e.first));
    assert(g3.contains(e.second));
    assert(g3.contains(e.first, e.second));
    assert( !g3.contains(e.second, e.first) );
  }
  assert( !g3.contains(8,2) );
  assert( !g3.contains(8) );
}

void test_vertices() {
  cout << "Testing vertices(): " << endl;
  graph g;
  boost::array<V, 11> verts = {{0,1,2,3,4,5,6,7,8,9,10}};
  std::map<V,V> vert_map;
  foreach(V v, verts) {
    vert_map[v] = v;
    g.add_vertex(v,v);
  }
  foreach(V v, g.vertices()) {
    assert(vert_map.count(v) == 1);
    assert(g[v] == vert_map[v]);
    vert_map.erase(v);
  }
  assert(vert_map.empty());
}

void test_edges() {
  cout << "Testing edges(): " << endl;
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
    assert(data.count(make_pair(e.source(), e.target())) == 1);
    assert(data[make_pair(e.source(), e.target())] == g[e]);
    data.erase(make_pair(e.source(), e.target()));
  }
  assert(data.empty());
}

void test_parents() {
  cout << "Testing neighbors(): " << endl;
  boost::array<E, 12> edges = 
    {{E(0, 2), E(4, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(1, 4),
      E(2, 6), E(3, 5), E(7, 4), E(5, 8)}};
  graph g(edges);
  boost::array<V, 3> verts = {{1,3,7}};
  std::set<V> vert_set;
  vert_set.insert(verts.begin(), verts.end());
  foreach(V v, g.parents(4)) {
    assert(vert_set.count(v) == 1);
    vert_set.erase(v);
  }
  assert(vert_set.empty());
}


void test_children() {
  cout << "Testing neighbors(): " << endl;
  boost::array<E, 12> edges = 
    {{E(0, 2), E(4, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(4, 7), E(5, 8)}};
  graph g(edges);
  boost::array<V, 4> verts = {{0,1,7,9}};
  std::set<V> vert_set;
  vert_set.insert(verts.begin(), verts.end());
  foreach(V v, g.children(4)) {
    assert(vert_set.count(v) == 1);
    vert_set.erase(v);
  }
  assert(vert_set.empty());
}



void test_in_edges() {
  cout << "Testing in_edges(): " << endl;
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
    assert(edge_values.count(e) == 1);
    assert(g[edge] == edge_values[e]);
    edge_values.erase(e);
  }
  assert(edge_values.empty());
}

void test_out_edges() {
  cout << "Testing in_edges(): " << endl;
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
    assert(edge_values.count(e) == 1);
    assert(g[edge] == edge_values[e]);
    edge_values.erase(e);
  }
  assert(edge_values.empty());
}


void test_contains() {
  cout << "Testing contains(): " << endl;
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g(edges);
  assert(g.contains(0) & g.contains(1) & g.contains(2) & g.contains(3) &
         g.contains(4) & g.contains(5) & g.contains(6) & g.contains(7) &
         g.contains(8) & g.contains(9) & g.contains(10) & !g.contains(11));
  foreach(E e, edges) {
    assert(g.contains(e.first, e.second) & !g.contains(e.second, e.first));
    assert(g.contains(g.get_edge(e.first,e.second)));
  }
  assert(!g.contains(0,3) & !g.contains(2, 10));
}


void test_get_edge() {
  cout << "Testing get_edge(): " << endl;
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
    assert(g.get_edge(e.first, e.second).source() == e.first);
    assert(g.get_edge(e.first, e.second).target() == e.second);
    assert(g[g.get_edge(e.first, e.second)] == data[e]);
  }
}


void test_degree() {
  cout << "Testing in degree" << endl;
  graph g;
  g.add_vertex(2,3);
  assert(g.in_degree(2) == 0);
  assert(g.out_degree(2) == 0);
  assert(g.degree(2) == 0);

  g.add_edge(1,2,3);
  assert(g.in_degree(2) == 1);
  assert(g.in_degree(1) == 0);
  assert(g.out_degree(1) == 1);
  assert(g.out_degree(2) == 0);
  assert(g.degree(1) == 1);
  assert(g.degree(2) == 1);

  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g2(edges);
  assert(g2.in_degree(3) == 3 &&
	 g2.out_degree(4) == 2 &&
	 g2.in_degree(3) + g2.out_degree(3) == g2.degree(3));
}


void test_num() {
  cout << "Testing num_vertices() and num_edges(): " << endl;
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g(edges);
  assert(g.num_vertices() == 11);
  assert(g.num_edges() == 12);
  g.clear_in_edges(3);
  assert(g.num_vertices() == 11);
  assert(g.num_edges() == 9);
  g.clear_out_edges(3);
  assert(g.num_vertices() == 11);
  assert(g.num_edges() == 6);

  g.remove_vertex(3);
  assert(g.num_vertices() == 10);
  assert(g.num_edges() == 6);
  g.remove_vertex(0);
  assert(g.num_vertices() == 9);
  assert(g.num_edges() == 4);
}



void test_large() {
  size_t n = 100000;
  size_t m = 100;
  
  cout << "Beginning Large Test" << endl;
  directed_graph<size_t, size_t, size_t> g;
  for(size_t u = 0; u < n; u++) {
    g.add_vertex(u,u);
    for(size_t v = 0; v < m; v++) {
      g.add_vertex(v,v);
      g.add_edge(u,v, u * v);
    } 
  }
}

void test_partial_vertex_order() {
  boost::array<E, 6> edges = 
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), 
      E(2, 3), E(3, 4)}};
  graph g(edges);
  cout << "Testing directed_partial_vertex_order(): " << endl;
  cout << "\t" << directed_partial_vertex_order(g) << endl;
}

int main(int argc, char** argv) {
  std::cout << "Testing Directed Graph" << std::endl;
  test_directed_edge();
  test_constructors();
  test_vertices();
  test_edges();
  test_parents();
  test_children();
  test_in_edges();
  test_out_edges();
  test_contains();
  test_get_edge();
  test_degree();
  test_num();

  // test_large();
  // test_partial_vertex_order();

  cout << "====================================================" << endl;
  cout << "PASSED ALL TESTS!" << endl;

  return(EXIT_SUCCESS);
}


