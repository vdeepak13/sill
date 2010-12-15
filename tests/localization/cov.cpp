#include <iostream>
#include <fstream>
#include <list>
#include <algorithm>

#include <prl/global.hpp>
#include <prl/range/numeric.hpp>
#include <prl/math/bindings/wm4.hpp>
#include <prl/stl_io.hpp>

#include <prl/range/algorithm.hpp>

typedef prl::math::bindings::wm4_kernel<double,3> kernel;
typedef kernel::matrix matrix_type;
typedef kernel::vector vector_type;

int main(int argc, char* argv[]) {
  using namespace std;
  using namespace boost;
  using namespace prl;
  
  assert(argc==2);
  ifstream is(argv[1]);
  assert(is.is_open());

  list<kernel::vector> pts;
  while (!is.eof()) {
    vector_type v;
    is >> v[0] >> v[1] >> v[2];
    if (!is.fail()) pts.push_back(v);
  }

  cout << "Points (" << pts.size() << ")" << pts << endl;

  cout << cov<matrix_type>(pts,pts,false) << endl;
}
