

#include <sill/math/sparse_linear_algebra/sparse_linear_algebra.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  std::cout << "Test of vector-matrix ops\n"
            << "------------------------------------------" << std::endl;

  double c = 3;
  std::cout << "c: " << c << std::endl;

  size_t n = 10;
  size_t k = n / 3;
  std::vector<size_t> indices;
  std::vector<double> values;
  for (size_t i = 0; i < k; ++i) {
    indices.push_back(2 * i);
    values.push_back(2 * i + 1);
  }
  sparse_vector<double> sv(n, indices, values);
  std::cout << "sv: " << sv << std::endl;

  vector<double> dv(n);
  for (size_t j = 0; j < n / 2; ++j) {
    dv[2 * j] = 2 * j + 2;
  }
  std::cout << "dv: " << dv << std::endl;

  size_t m = 3;
  matrix<double> dm(m,n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      dm(i,j) = i * m + j + 1;
    }
  }
  std::cout << "dm: " << dm << std::endl;

  std::cout << std::endl;

  // Vector-scalar ops
  {
    std::cout << "c * sv = " << (c * sv) << std::endl;
  }

  // Vector-vector ops
  {
    vector<double> tmp_dv(dv);
    tmp_dv += sv;
    std::cout << "dv += sv --> " << tmp_dv << std::endl;

    sparse_vector<double> tmp_sv(sv);
    tmp_sv -= dv;
    std::cout << "sv -= dv --> " << tmp_sv << std::endl;

    tmp_sv = sv;
    tmp_sv /= dv;
    std::cout << "sv /= dv --> " << tmp_sv << std::endl;

    {
      double tmp = dot(dv,sv);
      std::cout << "dot(dv, sv) = " << tmp << std::endl;

      assert(tmp == dot(dv, make_sparse_vector_view(sv)));
    }

    std::cout << "outer_product(dv, sv) =\n" << outer_product(dv,sv);

    elem_mult_out(sv, sv, tmp_sv);
    std::cout << "elem_mult(sv, sv) = " << tmp_sv << std::endl;
  }

  // Matrix-vector ops
  {
    std::cout << "dm * sv = " << (dm * sv) << std::endl;
  }

  // Matrix-matrix ops
  {
    matrix<double> tmp_dm(dm);
    vector<double> tmp_dv(dv(irange(0,dm.size1())));
    tmp_dm += outer_product(tmp_dv,sv);
    std::cout << "dm += outer_product(" << tmp_dv << ",sv) -->\n" << tmp_dm
              << std::endl;
  }

  std::cout << std::endl;

  return 0;

}

#include <sill/macros_undef.hpp>
