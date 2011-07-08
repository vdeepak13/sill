#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/mixture.hpp>

int main() {
  using namespace sill;
  using namespace std;

  universe u;
  vector_domain args;
  args.insert(u.new_vector_variable("x", 2));

  mixture_gaussian mix(2, args);
  mix[0] = moment_gaussian(make_vector(args), "1 2", eye(2,2));
  mix[1] = moment_gaussian(make_vector(args), "-1 -2", eye(2,2));
  cout << mix << endl;

  cout << project(mix) << endl;
}
