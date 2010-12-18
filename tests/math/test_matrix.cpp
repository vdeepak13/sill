#include <iostream>

#include <sill/math/matrix.hpp>
#include <sill/math/vector.hpp>
#include <sill/math/linear_algebra.hpp>

// test invert

int main() {
  using namespace sill;
  using namespace std;

  mat a = "1 2 3; 4 5 6; 7 8 9";
  mat b = "1 3; 4 6";
  mat b2 = "-1 -2; -3 -4";
  mat a2 = "-1 2 -2; -3 5 -4; 7 8 9";
  ivec i01 = "0 1";
  ivec i02 = "0 2";
  
  assert(a(i01, i02) == b);
  a.set_submatrix(i01, i02, b2);
  assert(a == a2);
  
  cout << a(i01, i02) << endl;
  a.add_submatrix(i01, i02, b2);
  cout << a(i01, i02) << endl;
  cout << a(irange(0,1), irange(0,2)) << endl;

  mat m = zeros(2,2);
  mat minv;
  cout << inv(m) << endl;
  bool result = inv(m, minv);
  cout << minv << endl;
  assert(!result);
}
