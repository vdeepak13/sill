#include <iostream>

#include <sill/math/vector.hpp>

int main() {
  using namespace sill;
  using namespace std;

  vec a = "1 2 3 4 5";
  vec b = "2 4";
  vec b2 = "-1 -3";
  vec a2 = "1 -1 3 -3 5";
  ivec i = "1 3";
  
  assert(a(i) == b);
  a.set_subvector(i, b2);
  assert(a == a2);
  
  cout << a(i) << endl;
  a.add_subvector(i, b2);
  cout << a(i) << endl;
  cout << a(irange(0,2)) << endl;
}
