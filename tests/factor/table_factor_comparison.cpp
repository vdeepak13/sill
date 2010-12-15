#include <iostream>
#include <string>
#include <iterator>
#include <cmath>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>

#include <prl/base/universe.hpp>
#include <prl/math/gdl_enum.hpp>
#include <prl/factor/table_factor.hpp>

int main(int argc, char** argv) {
  using namespace prl;
  using namespace boost;
  using namespace std;
  /*
  boost::mt19937 rng;
  boost::uniform_01<boost::mt19937, double> unif01(rng); */

  // Create a universe.
  universe u;

  // Create some variables and an argument set.
  finite_variable* x = u.new_finite_variable(2);
  finite_variable* y = u.new_finite_variable(2);
  finite_variable* z = u.new_finite_variable(2);
  finite_var_vector xy = make_vector(x, y);
  finite_var_vector yx = make_vector(y, x);
  finite_var_vector xz = make_vector(x, z);

  array<double, 4> v1 = {{ 1, 2, 3, 4 }};
  array<double, 4> v2 = {{ 1, 3, 2, 4 }};
  array<double, 4> v3 = {{ 1, 2, 3, 0 }};

  table_factor f1 = make_dense_table_factor(xy, v1);
  table_factor f2 = make_dense_table_factor(yx, v2);
  table_factor f3 = make_dense_table_factor(xy, v3);
  table_factor f4 = make_dense_table_factor(xz, v1);
  table_factor f5 = f1;

  assert(f1 == f2);
  assert(f1 != f3);
  cout << "f1==f2: " << (f1==f2) << endl;
  cout << "f2==f3: " << (f2==f3) << endl;
  cout << "f3==f4: " << (f3==f4) << endl;
  cout << "f1==f5: " << (f1==f5) << endl;
  cout << "f2==f5: " << (f2==f5) << endl;
  cout << "f2<f3: " << (f2<f3) << endl;
  cout << "f2>f3: " << (f3<f2) << endl;
  cout << "f3<f4: " << (f3<f4) << endl;
  cout << "f2<f4: " << (f3<f4) << endl;
}
