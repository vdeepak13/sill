#include <iostream>
#include <fstream>
#include <list>
#include <algorithm>

#include <sill/global.hpp>
#include <sill/geometry/alignment.hpp>
#include <sill/math/bindings/wm4.hpp>
#include <sill/stl_io.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/function.hpp>

#include <sill/range/algorithm.hpp>

typedef sill::math::bindings::wm4_kernel<double,3> kernel;
typedef Wm4::Quaternion<double> quaternion;
typedef kernel::matrix matrix_type;
typedef kernel::vector vector_type;

template <typename Engine, typename Distribution>
boost::variate_generator<Engine, Distribution>
make_variate_generator(Engine e, Distribution d) {
  return boost::variate_generator<Engine,Distribution>(e, d);
}

int main(int argc, char* argv[]) {
  using namespace std;
  using namespace boost;
  using namespace sill;
  using namespace sill::geometry;
  
  assert(argc==3);
  size_t count = lexical_cast<size_t>(argv[1]);
  double noise = lexical_cast<double>(argv[2]); 
  
  mt19937 rng;
  function0<double> uniform = make_variate_generator(&rng, uniform_real<>());
  
  quaternion q(uniform(), uniform(), uniform(), uniform());
  q.Normalize();
  matrix_type r(q); // the rotation matrix
  vector_type t(uniform(), uniform(), uniform());
  cout << "Random transform" << endl;
  cout << " q = " << q << endl;
  cout << " r = " << r << endl;
  cout << " t = " << t << endl;
  
  list<kernel::vector> source, target;
  for(size_t i=0;i<count;i++) {
    vector_type v, w;
    sill::generate(v, uniform);
    sill::generate(w, uniform); 
    source.push_back(v);
    //target.push_back(v);
    target.push_back(prod(r,v) + t + w*noise - noise/2);
  }
  cout << "Generated " << count << " 3D points." << endl;
  
  // cout << "Source = " << source << endl;
  // cout << "Target = " << target << endl;

  // Compute the least-squares rigid body transform
  matrix_type rotation;
  vector_type translation;
  tie(rotation,translation) = ralign<kernel>(source, target);
  cout << "Computed transform " << endl;
  cout << " q = " << quaternion(rotation) << endl;
  cout << " r = " << rotation << endl;
  cout << " t = " << translation << endl << endl;
  cout << "Error " << endl;
  cout << "|q-qh|_1 = " << norm_1(t-translation) << endl;
  cout << "|r-rh|_F = " << norm_frobenius(r-rotation) << endl;
}
