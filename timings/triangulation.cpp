#include <boost/array.hpp>
#include <boost/timer.hpp>

#include <iostream>
#include <functional>
#include <set>

#include <sill/graph/grid_graph.hpp>
#include <sill/stl_io.hpp>
#include <sill/graph/triangulation.hpp>
#include <sill/graph/min_degree_strategy.hpp>
#include <sill/graph/min_fill_strategy.hpp>
#include <sill/graph/undirected_graph.hpp>

#include <sill/macros_def.hpp>

int main() {
  using namespace std;
  using namespace sill;

  typedef undirected_graph<size_t> graph_type;

  // Build the graph.  Note that this graph must have no self-loops or
  // parallel edges.
  typedef std::pair<size_t, size_t> E;
  boost::array<E, 8> edges =
    {{E(0, 2), E(1, 2), E(1, 3), E(1, 5), E(2, 3), E(3, 4), E(4, 0), E(4, 1)}};
  graph_type g(edges);

  // Write the graph out.
  cout << "Graph: " << endl;
  cout << g << endl;

  // Triangulate the graph.
  typedef std::set<size_t> clique_type;
  std::list<clique_type> cliques;
  triangulate(g, std::front_inserter(cliques), min_degree_strategy(), false);
  
  // Report the cliques:
  cout << "Cliques: " << cliques << endl;

  // Triangulate lattices of varying size using min-degree and min-fill.
  for (size_t height = 1; height < 10; ++height) {
    for (size_t width = 32; width <= 1024; width *= 2) {
      // Build a lattice graph.
      graph_type lattice;
      make_grid_graph(height, width, lattice);

      // Create a copy to triangulate
      // (remember that triangulation is a destructive operation.)
      graph_type lattice_copy(lattice);
      boost::timer t;

      // Triangulate the graph using min-degree triangulation.
      cliques.clear();
      t.restart();
      triangulate(lattice_copy, std::front_inserter(cliques),
		  min_degree_strategy());
      double time = t.elapsed();

      // Compute the tree_width.
      size_t tree_width = 0;
      foreach(const clique_type& clique, cliques) 
	tree_width = std::max(tree_width, clique.size() - 1);

      // Report the speed.
      cout << "Computed tree width " << tree_width 
	   << " triangulation of " << height << "x" << width 
	   << " lattice in "
	   << time << "s using min-degree." << endl;


      // Triangulate the graph using min-fill triangulation.
      cliques.clear();
      t.restart();
      triangulate(lattice, std::front_inserter(cliques), min_fill_strategy());
      time = t.elapsed();

      // Compute the tree_width.
      tree_width = 0;
      foreach(const clique_type& clique, cliques) 
	tree_width = std::max(tree_width, clique.size() - 1);

      // Report the speed.
      cout << "Computed tree width " << tree_width 
	   << " triangulation of " << height << "x" << width 
	   << " lattice in "
	   << time << "s using min-fill." << endl;
    }
  }

  return EXIT_SUCCESS;
}
