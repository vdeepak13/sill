#include <iostream>

#include <prl/math/function/logistic_discrete.hpp>

int main() {
  using namespace std;
  using namespace prl;

  logistic_discrete f("1 2 3; 4 5 6", 1);
  assert(abs(f(ivec("0 2")) - 1/(1+exp(-8.0))) < 1e-10);
}
