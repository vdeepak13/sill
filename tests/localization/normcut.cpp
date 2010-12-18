#include <iostream>
#include <fstream>
#include <list>
#include <algorithm>
#include <iterator>

#include <sill/global.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/functional.hpp>
#include <sill/math/bindings/lapack.hpp>
#include <sill/graph/normalized_cut.hpp>
#include <sill/stl_io.hpp>

#include <sill/range/algorithm.hpp>

#include <sill/macros_def.hpp>

typedef sill::math::bindings::lapack_kernel<double> kernel;
typedef kernel::symmetric_matrix matrix_type;
typedef kernel::vector vector_type;
typedef std::pair<size_t,size_t> vertex_pair;

int main(int argc, char* argv[]) {
  using namespace std;
  using namespace boost;
  using namespace sill;
  
  assert(argc==2);
  ifstream is(argv[1]);
  assert(is.is_open());
  
  // Load a list of neighbors
  list<vertex_pair> edges;
  while (!is.eof()) { 
    size_t from, to;
    is >> from >> to;
    if (!is.fail()) edges.push_back(make_pair(from, to));
  }
  size_t n = max(max(edges | sill::transformed(pair_first<size_t,size_t>())),
		 max(edges | sill::transformed(pair_second<size_t,size_t>())));
  cout << "Loaded " << edges.size() << " neighbors; n = " << n << endl;

  // Create the adjacency matrix
  matrix_type w(n,n);
  foreach(vertex_pair e, edges) {
    w(e.first-1, e.second-1) = 1;
    w(e.second-1, e.first-1) = 1;
  }
  
  // Compute & store the relaxed solution
  vector_type relaxed = sill::graph::normalized_cut<kernel>(w);
  cout << relaxed << endl;
  list<size_t> left, right;
  find_indices(relaxed <=0, back_inserter(left));
  find_indices(relaxed > 0, back_inserter(right));
  cout << "Left: " << left << endl;
  cout << "Right: " << right << endl;
}
