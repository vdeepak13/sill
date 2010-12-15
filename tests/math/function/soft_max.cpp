#include <iostream>

#include <prl/math/function/soft_max.hpp>

int main() {
  using namespace prl;
  using namespace std;
    
  soft_max sm("1;-1", "1 1");
  cout << sm(ones(1)) << endl;
  // ought to be [exp(2) 1] normalized \approx 0.8808 0.1192
}
