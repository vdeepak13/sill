#include <iostream>
#include <functional>
#include <set>

#include <boost/array.hpp>
#include <boost/timer.hpp>

#include <prl/functional.hpp>
#include <prl/graph/grid_graphs.hpp>
#include <prl/graph/undirected_graph.hpp>
#include <prl/graph/min_degree_strategy.hpp>
#include <prl/graph/min_fill_strategy.hpp>
#include <prl/model/junction_tree.hpp>

/**
 * \file junction_tree.cpp Junction Tree test
 */
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

  // Print the graph.
  cout << "Original graph: " << endl;
//  boost::print_graph(g);
  cout << g;

  // Build a junction tree for the graph.
  junction_tree<size_t> jt(g, min_degree_strategy());
  jt.check_validity();

  // Print the junction tree.
  cout << "Junction tree: " << jt << endl;
  cout << jt << endl;

  // Copy the junction tree and check validity of original and copy.
  cout << "Copying junction tree:" << endl;
  junction_tree<size_t> jt2(jt);
  cout << "  Checking validity of the original ... ";
  jt.check_validity();
  cout << "OK" << endl;
  cout << "  Checking validity of the copy ... ";
  jt2.check_validity();
  cout << "OK" << endl;

  // Print the subtree cover for the set {0, 5}.
  std::set<size_t> set; set.insert(5); set.insert(0);
  cout << "After marking subtree cover for " << set << ": " << endl;
  jt.mark_subtree_cover(set, true);
  cout << jt << endl;

  // Print the subtree cover for the set {1}.
  set.clear(); set.insert(1);
  cout << "After marking subtree cover for " << set << ": " << endl;
  jt.mark_subtree_cover(set, true);
  cout << jt << endl;

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
