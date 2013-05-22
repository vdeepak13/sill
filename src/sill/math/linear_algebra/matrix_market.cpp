
#include <sill/math/linear_algebra/matrix_market.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  matrix_market_info::matrix_market_info(const std::string& filepath) {

    using namespace mm;
    FILE *f = NULL;
    MM_typecode matcode;
    int M, N, nz;

    // Read header.
    //------------------------------------------------------

    if ((f = fopen(filepath.c_str(), "r")) == NULL) {
      std::cerr << "matrix_market_info constructor failed to open file: "
                << filepath << std::endl;
      assert(false);
      return;
    }

    if (mm_read_banner(f, &matcode) != 0) {
      std::cerr
        << "matrix_market_info could not process Matrix Market banner in file: "
        << filepath << std::endl;
      assert(false);
      return;
    }

    is_matrix = mm_is_matrix(matcode);

    is_sparse = mm_is_sparse(matcode);
    is_coordinate = mm_is_coordinate(matcode);
    is_dense = mm_is_dense(matcode);
    is_array = mm_is_array(matcode);

    is_complex = mm_is_complex(matcode);
    is_real = mm_is_real(matcode);
    is_pattern = mm_is_pattern(matcode);
    is_integer = mm_is_integer(matcode);

    is_symmetric = mm_is_symmetric(matcode);
    is_general = mm_is_general(matcode);
    is_skew = mm_is_skew(matcode);
    is_hermitian = mm_is_hermitian(matcode);

    is_valid = mm_is_valid(matcode);

    // Find out size of matrix.
    if (mm_is_sparse(matcode)) {
      if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0) {
        impl::errormsg("load_matrix_market failed to read matrix size.",
                       filepath);
        return;
      }
    } else {
      if (mm_read_mtx_array_size(f, &M, &N) !=0) {
        impl::errormsg("load_matrix_market failed to read matrix size.",
                       filepath);
        return;
      }
      nz = M * N;
    }

    num_rows = (size_t)M;
    num_cols = (size_t)N;
    num_non_zeros = (size_t)nz;

    fclose(f);
    f = NULL;
  } // matrix_market_info::matrix_market_info(const std::string& filepath)

} // namespace sill

#include <sill/macros_undef.hpp>
