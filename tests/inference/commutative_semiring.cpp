// TODO: convert the old gdl test to commutative_semiring

#include <cassert>
#include <iostream>
#include <vector>

#include <sill/math/gdl_enum.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/range/numeric.hpp>

#include <pstade/oven/counting.hpp>
#include <pstade/oven/copied.hpp>

int main(int argc, char** argv) {
  using namespace sill;
  using namespace pstade::oven;
  using namespace std;

  // Create a collection of real numbers {0, 1, 2, 3, 4}
  vector<double> v = counting(0, 5) | copied;

  // Compute their sum, product, max, and min explicitly.
  double sum = sill::sum(v), prod = sill::prod(v);
  double max = sill::max(v), min = sill::min(v);

  cout << sum << " " << prod << " " << max << " " << min << endl;

  // Ensure these agree with the GDL computations.
  assert(sum  == cross_all(v, sum_product));
  assert(prod == dot_all(v, sum_product));
  assert(max  == cross_all(v, max_product));
  assert(prod == dot_all(v, max_product));
  assert(min  == cross_all(v, min_sum));
  assert(sum  == dot_all(v, min_sum));
  assert(max  == cross_all(v, max_sum));
  assert(sum  == dot_all(v, max_sum));

  // Do the same for the Boolean commutative semiring.
  bool b[2] = { false, true };
  vector<bool> vb(b, b+2);
  assert(true == cross_all(vb, boolean));
  assert(false == dot_all(vb, boolean));

  // Return success.
  return EXIT_SUCCESS;
}
