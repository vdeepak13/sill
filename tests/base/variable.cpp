#include <iostream>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/universe.hpp>

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Create a universe.
  universe u;

  // Create some variables.
  finite_variable* x = u.new_finite_variable("x", 3);
  vector_variable* y = u.new_vector_variable("y", 2);
  finite_variable* z = u.new_finite_variable(2);
  vector_variable* q = u.new_vector_variable(2);

  cout << x << endl;
  cout << y << endl;
  cout << z << endl;
  cout << q << endl;

  return EXIT_SUCCESS;
}
