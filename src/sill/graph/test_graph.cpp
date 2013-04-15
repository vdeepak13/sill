#include <sill/graph/directed_graph.hpp>
#include <sill/graph/undirected_graph.hpp>

using namespace sill;

template <typename Graph>
void test() {
  Graph g1, g2;
  g1.add_vertex(1, 0);
  g1.add_vertex(2, 1);
  g1.add_vertex(3, 2);
  g1.add_edge(1, 2);
  g1.add_edge(2, 3);
  
  g2.add_vertex(1, 0);
  g2.add_vertex(3, 2);
  g2.add_vertex(2, 1);
  g2.add_edge(1, 2);
  g2.add_edge(2, 3);

  assert(g1 == g2);

  Graph g3 = g2;
  g3[1] = -1;
  assert(g2 != g3);

  Graph g4 = g2;
  g4.remove_edge(2, 3);
  g4.add_edge(1, 3);
  assert(g2 != g4); 
}

int main() {
  test<directed_graph<int, int> >();
  test<directed_graph<int, int, int> >();
  test<undirected_graph<int, int> >();
  test<undirected_graph<int, int, int> >();
}
