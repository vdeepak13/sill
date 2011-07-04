#include <iostream>

#include <sill/math/function/logistic_discrete.hpp>

int main() {
  using namespace std;
  using namespace sill;

  logistic_discrete f("1 2 3; 4 5 6", 1);
  assert(abs(f(uvec("0 2")) - 1/(1+exp(-8.0))) < 1e-10);
}
