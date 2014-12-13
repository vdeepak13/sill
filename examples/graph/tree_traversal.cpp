#include <iostream>

#include <boost/array.hpp>

#include <sill/graph/undirected_graph.hpp>
#include <sill/graph/algorithm/tree_traversal.hpp>

struct write_edge_visitor {
  template <typename Graph>
  void operator()(typename Graph::edge e, Graph&) {
    std::cout << e << std::endl;
  }
};

struct vertex_printer {
  template <typename Graph,typename OutputStream>
  void operator()(typename Graph::vertex v,
                  const Graph&, OutputStream& out) {
    out << v;
  }
};

int main() {
  using namespace sill;
  using namespace std;

  typedef std::pair<size_t, size_t> E;
  boost::array<E, 5> edges = {{ E(0, 1), E(0, 2), E(1, 3), E(1, 5), E(1, 4) }};

  undirected_graph<size_t> g(edges);

  cout << "Pre-order traversal from 0:" << endl;
  pre_order_traversal(g, 0, write_edge_visitor());

  cout << "Post-order traversal from 0:" << endl;
  post_order_traversal(g, 0, write_edge_visitor());

  print_tree(g, cout, vertex_printer());

  return EXIT_SUCCESS;
}
