#include <iostream>
#include <functional>
#include <set>

#include <boost/array.hpp>

#include <prl/functional.hpp>
#include <prl/graph/grid_graphs.hpp>
#include <prl/graph/undirected_graph.hpp>
#include <prl/graph/min_degree_strategy.hpp>
#include <prl/graph/min_fill_strategy.hpp>
#include <prl/model/learnt_junction_tree.hpp>


int main() {

  using namespace prl;
  using namespace std;

  typedef undirected_graph<size_t> graph_type;

  // Build the graph.  Note that this graph must have no self-loops or
  // parallel edges.
  typedef pair<size_t, size_t> E;
  boost::array<E, 8> edges =
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 5), E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  graph_type g(edges);

  // Build a junction tree for the graph.
  learnt_junction_tree<size_t> jt(g, min_degree_strategy());

  // Print the graph.
  cout << "Original jt:\n" << jt << endl;

  std::set<size_t> c0;
  c0.insert(0);
  learnt_junction_tree<size_t>::vertex v0(jt.find_clique_cover(c0));

  // Make some changes.
  cout << "Adding clique (0,6):" << endl;
  std::set<size_t> c06;
  c06.insert(0);
  c06.insert(6);
  learnt_junction_tree<size_t>::vertex v06(jt.add_clique(c06));
  cout << jt << endl;

  cout << "Adding edge between clique (0,6) and a clique with 0 in it:" << endl;
  jt.add_edge(v0,v06);
  cout << jt << endl;

  return EXIT_SUCCESS;
}
