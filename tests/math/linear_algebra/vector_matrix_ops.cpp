

#include <sill/math/linear_algebra/sparse_linear_algebra.hpp>

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

  vec dv(n);
  for (size_t j = 0; j < n / 2; ++j) {
    dv[2 * j] = 2 * j + 2;
  }
  std::cout << "dv: " << dv << std::endl;

  size_t m = 3;
  arma::Mat<double> dm(m,n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      dm(i,j) = i * m + j + 1;
    }
  }
  std::cout << "dm: " << dm << std::endl;

  coo_matrix<double> coomat(m, n, m*n/2);
  {
    size_t k = 0;
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        coomat.row_indices()[k] = i;
        coomat.col_indices()[k] = j;
        coomat.values()[k] = i * m + j + 1;
        ++k;
        if (k >= coomat.num_non_zeros())
          break;
      }
      if (k >= coomat.num_non_zeros())
        break;
    }
  }
  csc_matrix<double> cscmat(coomat);
  std::cout << "cscmat: " << cscmat << std::endl;

  std::cout << std::endl;

  // Vector-scalar ops
  {
    std::cout << "c * sv = " << (c * sv) << std::endl;
    std::cout << "sum(sv) = " << sum(sv) << std::endl;
  }

  // Vector-vector ops
  {
    vec tmp_dv(dv);
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
    std::cout << "sum(cscmat,0) = " << sum(cscmat,0) << std::endl;
    std::cout << "sum(cscmat,1) = " << sum(cscmat,1) << std::endl;
  }

  // Matrix-vector ops
  {
    vec tmp_dv(dm * sv);
    std::cout << "dm * sv = " << tmp_dv << std::endl;
    tmp_dv.zeros();
    sill::gemv(dm, sv, tmp_dv);
    std::cout << "tmp_dv from gemv(dm, sv, tmp_dv) = " << tmp_dv << std::endl;
  }

  // Matrix-matrix ops
  {
    arma::Mat<double> tmp_dm(dm);
    vec tmp_dv(dv.subvec(span(0,dm.n_rows-1)));
    tmp_dm += outer_product(tmp_dv,sv);
    std::cout << "dm += outer_product(" << tmp_dv << ",sv) -->\n" << tmp_dm
              << std::endl;
  }

  std::cout << std::endl;

  return 0;

}

#include <sill/macros_undef.hpp>
