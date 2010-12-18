#include <fstream>
#include <iostream>

#include <sill/serialization/serialize.hpp>

#include <sill/graph/directed_graph.hpp>
#include <sill/graph/undirected_graph.hpp>

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
  sill::oarchive oar(ofs);
  oar << g;
  ofs.close();

  std::ifstream ifs(filename, fstream::binary);
  sill::iarchive iar(ifs);
  iar >> h;
  ifs.close();

  cout << g << endl;
  cout << h << endl;
}

int main() {
  using namespace sill;
  test<directed_graph<int, std::string> >("test.bin");
  test<undirected_graph<int, std::string> >("test.bin");
}
