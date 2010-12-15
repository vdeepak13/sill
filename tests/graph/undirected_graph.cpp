#include <iostream>
#include <set>
#include <map>
#include <boost/array.hpp> 

#include <prl/graph/undirected_graph.hpp>

#include <prl/macros_def.hpp>


using namespace std;
using namespace prl;

typedef size_t V;
typedef undirected_graph<V, size_t, size_t> graph;
typedef std::pair<V, V> E;

void test_undirected_edge() {
  cout << "Testing Undirected Edge: " << endl;
  undirected_edge<V> e1, e2;
  cout << "Default edge: " << e1 << endl;
  assert(e1 == e2);
  assert(e1.source() == e2.source());
  assert(e1.source() == e1.target());
}

void test_constructors() {
  cout << "Testing Constructors: " << endl;
  cout << "Testing default constructor: " << endl;
  graph g1;
  assert(g1.empty());
  assert(g1.edges().first == g1.edges().second);
  assert(g1.vertices().first == g1.vertices().second);
  
  cout << "Testing edge list constructor: " << endl;
  boost::array<E, 8> edges = 
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 7), E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  graph g2(edges);
  foreach(E e, edges) {
    assert(g2.contains(e.first));
    assert(g2.contains(e.second));
    assert(g2.contains(e.first, e.second));
    assert(g2.contains(e.second, e.first));
  }
  assert( !g2.contains(8,2) );
  assert( !g2.contains(8) );

  cout << "Testing copy constructor: " << endl;
  graph g3 = g2;
  foreach(E e, edges) {
    assert(g3.contains(e.first));
    assert(g3.contains(e.second));
    assert(g3.contains(e.first, e.second));
    assert(g3.contains(e.second, e.first));
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
  foreach(graph::edge edge, g.edges()) {
    E e = make_pair(edge.source(), edge.target());
    E erev = make_pair(edge.target(), edge.source());
    assert( (data.count(e) == 1) ^ (data.count(erev) == 1) );
    if(data.count(erev) == 1) e = erev;
    assert(data[e] == g[edge]);
    data.erase(e);
  }
  assert(data.empty());
}


void test_neighbors() {
  cout << "Testing neighbors(): " << endl;
  boost::array<E, 12> edges = 
    {{E(0, 2), E(4, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(4, 7), E(5, 8)}};
  graph g(edges);
  boost::array<V, 5> verts = {{0,1,3,7,9}};
  std::set<V> neighbors;
  neighbors.insert(verts.begin(), verts.end());
  foreach(V v, g.neighbors(4))  {
    assert(neighbors.count(v) == 1);
    neighbors.erase(v);
  }
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
    if(e.first == u || e.second == u) {
      g.add_edge(e.first, e.second, i);
      if(e.first == u) e = make_pair(e.second, e.first);
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
    if(e.first == u  || e.second == u) {
      g.add_edge(e.first, e.second, i);
      if(e.second == u) e = make_pair(e.second, e.first);
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
    assert(g.contains(e.first, e.second) & g.contains(e.second, e.first));
    assert(g.contains(g.get_edge(e.first,e.second)) &
           g.contains(g.get_edge(e.second,e.first)));
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
    data[make_pair(e.second, e.first)] = i; 
    ++i;
  }
  foreach(E e, edges) {
    assert(g.get_edge(e.first, e.second).source() == e.first);
    assert(g.get_edge(e.first, e.second).target() == e.second);
    assert(g[g.get_edge(e.first, e.second)] == data[e]);
  }
}



void test_degree() {
  cout << "Testing degree(): " << endl;
  boost::array<E, 12> edges = 
    {{E(0, 2), E(1, 9), E(1, 3), E(1, 10), 
      E(2, 3), E(3, 4), E(4, 0), E(4, 1),
      E(2, 6), E(3, 5), E(7, 3), E(3, 8)}};
  graph g(edges);
  assert(g.in_degree(3) == g.out_degree(3) &&
         g.out_degree(3) == g.degree(3) &&
         g.degree(3) == 6);
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
  g.clear_edges(3);
  assert(g.num_vertices() == 11);
  assert(g.num_edges() == 6);
  g.remove_vertex(3);
  assert(g.num_vertices() == 10);
  assert(g.num_edges() == 6);
  g.remove_vertex(0);
  assert(g.num_vertices() == 9);
  assert(g.num_edges() == 4);
}


int main()
{
  test_undirected_edge();
  test_constructors();
  test_vertices();
  test_edges();
  test_neighbors();
  test_in_edges();
  test_out_edges();
  test_contains();
  test_get_edge();
  test_degree();
  test_num();

  cout << "====================================================" << endl;
  cout << "PASSED ALL TESTS!" << endl;
}
