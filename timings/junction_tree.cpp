#include <iostream>
#include <functional>
#include <set>

#include <boost/array.hpp>
#include <boost/timer.hpp>

#include <sill/functional.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/graph/undirected_graph.hpp>
#include <sill/graph/min_degree_strategy.hpp>
#include <sill/graph/min_fill_strategy.hpp>
#include <sill/model/junction_tree.hpp>

/**
 * \file junction_tree.cpp Junction Tree test
 */
int main() {

  using namespace sill;
  using namespace std;

  typedef undirected_graph<size_t> graph_type;

  for (size_t height = 1; height < 10; ++height) {
    for (size_t width = 32; width <= 1024; width *= 2) {
      // Build a lattice graph.
      graph_type lattice;
      make_grid_graph(height, width, lattice);

      // Create a copy to triangulate
      // (remember that triangulation is a destructive operation.)
      graph_type lattice_copy(lattice);

      // Build a junction tree for it using min-degree triangulation.
      double time;
      size_t tree_width;
      {
        boost::timer t;
        junction_tree<size_t> jt(lattice_copy, min_degree_strategy());
        time = t.elapsed();
        tree_width = jt.tree_width();

        // Report the speed.
        cout << "Built tree width " << tree_width
             << " junction tree for "
             << height << "x" << width
             << " lattice in "
             << time << "s using min-degree." << endl;
      }
      // Build a junction tree for it using min-fill triangulation.
      {
        boost::timer t;
        junction_tree<size_t> jt(lattice, min_fill_strategy());
        time = t.elapsed();
        tree_width = jt.tree_width();

        // Report the speed.
        cout << "Built tree width " << tree_width
             << " junction tree for "
             << height << "x" << width
             << " lattice in "
             << time << "s using min-fill." << endl;
      }
    }
  }

  return EXIT_SUCCESS;
}
