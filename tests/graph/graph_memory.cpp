#include <boost/lexical_cast.hpp>

#include <prl/graph/directed_graph.hpp>
#include <prl/graph/directed_multigraph.hpp>

// tests the memory usage of graphs
int main(int argc, char** argv) {
  using namespace std;
  using namespace prl;

  if (argc < 3) {
    cerr << "Usage: graph_size num_vertices num_neighbors" << endl;
    return -1;
  }

  size_t n = boost::lexical_cast<size_t>(argv[1]);
  size_t m = boost::lexical_cast<size_t>(argv[2]);

  //directed_multigraph<size_t> g;
  directed_graph<size_t> g;

  for(size_t i = 1; i <= n; i++) {
    g.add_vertex(i);
    for(size_t j = i-1; j > 0 && j+m > i; j--)
      g.add_edge(i, j);
  }

  cout << "Created a graph with " << n << " vertices and " << g.num_edges()
       << " edges" << endl;
  
  while(1);
}
