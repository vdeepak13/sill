
#include <sill/math/linear_algebra/sparse_linear_algebra.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

void test_sparse_vector();

int main(int argc, char** argv) {

  test_sparse_vector();

} // main

void test_sparse_vector() {

  size_t n = 10;
  size_t k = n / 3;
  vec dvec1(n);
  sparse_vector<double> svec1(n, k);
  for (size_t i = 0; i < k; ++i) {
    svec1.index(i) = 2 * i;
    svec1.value(i) = 2 * i + 1;
  }
  std::cout << "svec1: " << svec1 << std::endl;

}

#include <sill/macros_undef.hpp>
