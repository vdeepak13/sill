#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/multinomial_distribution.hpp>

#include <boost/random/mersenne_twister.hpp>

boost::mt19937 rng;

int main() {
  using namespace sill;
  using namespace std;

  vec count(4);
  cout << count << endl;

  multinomial_distribution dist("0.2 0.2 0.5 0.1");
  for(size_t i = 0; i < 10000; i++)
    count[dist(rng)]++;
  
  cout << count / sum(count) << endl;
}

