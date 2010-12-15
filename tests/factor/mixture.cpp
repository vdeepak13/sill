#include <iostream>

#include <prl/math/linear_algebra.hpp>
#include <prl/factor/moment_gaussian.hpp>
#include <prl/factor/canonical_gaussian.hpp>
#include <prl/factor/mixture.hpp>

int main() {
  using namespace prl;
  using namespace std;

  universe u;
  vector_var_vector v = u.new_vector_variables(2, 1);

  mixture_gaussian mix(2, vector_domain(v.begin(), v.end()));
  mix[0] = moment_gaussian(v, zeros(2), identity(2));
  mix[1] = moment_gaussian(v, ones(2), "2 1; 1 3");

  cout << mix << endl;

  mix.add_parameters(mix, 0.5);
  cout << mix << endl;
  
  mix.normalize();
  cout << mix << endl;

  return 0;
}
