#include <iostream>

#include <sill/math/linear_algebra/armadillo.hpp>

// test invert

int main() {
  using namespace sill;
  using namespace std;

  mat a = "1 2 3; 4 5 6; 7 8 9";
  mat b = "1 3; 4 6";
  mat b2 = "-1 -2; -3 -4";
  mat a2 = "-1 2 -2; -3 5 -4; 7 8 9";
  uvec i01 = "0 1";
  uvec i02 = "0 2";
  
  assert(accu(a(i01, i02) == b) == b.n_elem);
  a(i01, i02) = b2;
  assert(equal(a, a2));
  
  cout << a(i01, i02) << endl;
  a(i01, i02) += b2;
  cout << a(i01, i02) << endl;
  cout << a(span(0,0), span(0,1)) << endl;

  mat m = zeros(2,2);
  m(0,0) = 2;
  m(1,1) = 3;
  mat minv;
  cout << inv(m) << endl;
  bool result = inv(minv,m);
  cout << minv << endl;
  assert(result);
}
