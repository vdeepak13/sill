
#include <sill/math/linear_algebra/sparse_linear_algebra.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
using namespace std;

int main(int argc, char** argv) {

  cout << "Test of sparse_vector\n"
       << "------------------------------------------" << endl;

  size_t n = 10;
  size_t k = n / 3;
  std::vector<size_t> indices;
  std::vector<double> values;
  for (size_t i = 0; i < k; ++i) {
    indices.push_back(2 * i);
    values.push_back(2 * i + 1);
  }

  // Constructors.
  sparse_vector<double> v1;
  sparse_vector<double> v2(n);
  sparse_vector<double> v3(n, k);
  sparse_vector<double> v4(n, indices, values);
  v1.reset(n, indices, values);

  for (size_t i = 0; i < k; ++i) {
    v3.index(i) = 2 * i;
    v3.value(i) = 2 * i + 1;
  }

  // Comparisons.
  assert(v1 == v4);
  assert(v2 != v4);
  v1[0] -= 1;
  assert(v1 < v4);
  assert(v3 == v4);

  // Operations
  {
    double c = 3;
    cout << v1 << " * " << c << " = ";
    sparse_vector<double> tmp_v1(v1);
    tmp_v1 *= c;
    cout << tmp_v1 << endl;
  }

  // Methods to support optimization routines
  {
    sparse_vector<double> tmp_v1(v1);
    cout << "Element-wise square of " << v1 << " = ";
    tmp_v1.elem_mult(tmp_v1);
    cout << tmp_v1 << endl;
    cout << "L1norm(" << v1 << ") = " << v1.L1norm() << endl;
    cout << "L2norm(" << v1 << ") = " << v1.L2norm() << endl;
  }

  cout << endl;

  return 0;

}

#include <sill/macros_undef.hpp>
