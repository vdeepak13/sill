#include <iostream>
#include <string>
#include <iterator>
#include <cmath>
#include <prl/math/gdl_enum.hpp>
#include <prl/factor/constant_factor.hpp>
#include <prl/copy_ptr.hpp>

int main(int argc, char** argv) {

  using namespace prl;
  using namespace std;

  // Create a constant factor with no arguments
  constant_factor f(1.0);
  cout << "f=" << f << endl;

  // Create another constant factor with no arguments.
  constant_factor g(2.0);
  cout << "g=" << g << endl;

  // Multiply them together.
  constant_factor fg = combine(f, g, product_op); // is operator= invoked?
  cout << "fg=" << fg << endl;

  // Collapse the product down.
  constant_factor h = fg.collapse(finite_domain(), sum_op);
  cout << "sum(f)=" << h << endl;

  return EXIT_SUCCESS;
}
