#include <fstream>
#include <iostream>

#include <prl/serialization/serialize.hpp>

#include <prl/graph/directed_graph.hpp>
#include <prl/graph/undirected_graph.hpp>

template <typename Graph>
void test(const char* filename) {
  using namespace std;
  Graph g, h;
  g.add_vertex(1, "hello");
  g.add_vertex(2, "bye");
  g.add_vertex(3, "maybe");
  g.add_edge(1, 2);
  g.add_edge(2, 3);

  std::ofstream ofs(filename, fstream::binary);
  prl::oarchive oar(ofs);
  oar << g;
  ofs.close();

  std::ifstream ifs(filename, fstream::binary);
  prl::iarchive iar(ifs);
  iar >> h;
  ifs.close();

  cout << g << endl;
  cout << h << endl;
}

int main() {
  using namespace prl;
  test<directed_graph<int, std::string> >("test.bin");
  test<undirected_graph<int, std::string> >("test.bin");
}
