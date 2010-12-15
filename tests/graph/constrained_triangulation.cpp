#include <iostream>

#include <prl/global.hpp>
#include <prl/functional.hpp>
#include <prl/graph/triangulation.hpp>
#include <prl/graph/undirected_graph.hpp>
#include <prl/graph/grid_graphs.hpp>
#include <prl/graph/min_degree_strategy.hpp>
#include <prl/graph/constrained_elim_strategy.hpp>
#include <prl/model/junction_tree.hpp>


//! A function object that extracts the elimination priority from a graph
//! If needed, we could make this functor specific for graph_type below
struct elim_priority_functor {
  typedef size_t result_type;
  template <typename Graph>
  size_t operator()(typename Graph::vertex v, const Graph& graph) {
    return graph[v];
  }
};

int main() {

  using namespace prl;
  using namespace std;

  // The graph type.  Each vertex is annotated with the elimination priority
  typedef undirected_graph<size_t, size_t> graph_type;

  // Build a 2 x n lattice.
  size_t n = 5;
  graph_type lattice;

  // Add the vertices and prioritize their elimination so that
  // vertices on the top row (indexes 5-9) have lower elimination
  // priority than vertices on the bottom row (indexes 0-4).
  boost::multi_array<size_t,2> v = make_grid_graph(2, n, lattice);
  for(size_t i = 0; i < 2; i++)
    for(size_t j = 0; j < n; j++)
      lattice[v[i][j]] = i;

  // Write the graph out.
  cout << "Graph: " << endl;
  cout << lattice << endl;

  // Create a constrained elimination strategy (using min-degree as
  // the secondary strategy).
  constrained_elim_strategy<elim_priority_functor, min_degree_strategy> s;

  // Create a junction tree using this elimination strategy.  (If we
  // imagine vertices 0-4 as being discrete and vertices 5-9 as
  // continuous random variables, then this creates a strongly-rooted
  // junction tree.)
  junction_tree<size_t> jt(lattice, s);

  // Report the junction tree.
  cout << "Junction tree: " << endl << jt << endl;

  return EXIT_SUCCESS;
}

/*

Junction tree:
v
({0 5 6}  invalid)
({4 8 9}  invalid)
({3 4 7 8}  invalid)
({0 1 6 7}  invalid)
({0 1 2 3 4 7}  invalid)
e
2 4  ({3 4 7}  invalid)
4 3  ({0 1 7}  invalid)
0 3  ({0 6}  invalid)
2 1  ({4 8}  invalid)

*/
