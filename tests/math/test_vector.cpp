#include <iostream>

#include <sill/math/linear_algebra/armadillo.hpp>

int main() {
  using namespace sill;
  using namespace std;

  vec a = "1 2 3 4 5";
  vec b = "2 4";
  vec b2 = "-1 -3";
  vec a2 = "1 -1 3 -3 5";
  uvec i = "1 3";
  
  assert(accu(a(i) == b) == b.size());
  a.subvec(i) = b2;
  assert(accu(a == a2) == a.size());
  
  cout << a(i) << endl;
  a.subvec(i) += b2;
  cout << a(i) << endl;
  cout << a.subvec(span(0,1)) << endl;
}
