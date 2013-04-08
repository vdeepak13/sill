#ifndef SILL_MATRIX_MARKET_HPP
#define SILL_MATRIX_MARKET_HPP

#include <iostream>

#include <sill/math/linear_algebra/mmio.hpp>
#include <sill/math/linear_algebra/sparse_linear_algebra.hpp>

/**
 * \file matrix_market.hpp  Functions for working with Matrix Market formats.
 *
 * For more info, see http://math.nist.gov/MatrixMarket/mmio-c.html
 */

namespace sill {

  struct matrix_market_info {
    bool is_matrix;

    bool is_sparse;
    bool is_coordinate;
    bool is_dense;
    bool is_array;

    bool is_complex;
    bool is_real;
    bool is_pattern;
    bool is_integer;

    bool is_symmetric;
    bool is_general;
    bool is_skew;
    bool is_hermitian;

    bool is_valid;

    size_t num_rows;
    size_t num_cols;
    size_t num_non_zeros;

    //! Loads typecode from Matrix Market file (or sets is_valid==false).
    matrix_market_info(const std::string& filepath);

  }; // struct matrix_market_info

  namespace impl {

    //! @return 1 for error
    inline int errormsg(const std::string& msg, const std::string& filepath) {
      std::cerr << msg << "\n"
                << "\t in file: " << filepath << std::endl;
      assert(false);
      return 1;
    }

  } // namespace impl

  /**
   * Loads a matrix in Matrix Market format.
   * @return  0: success, 1: failure
   */
  template <typename T, typename I>
  int load_matrix_market(coo_matrix<T,I>& m, const std::string& filepath) {

    matrix_market_info mm_info(filepath);

    m.resize(mm_info.num_rows, mm_info.num_cols, mm_info.num_non_zeros);

    std::ifstream fin(filepath.c_str());
    assert(fin.good());
    std::string line;

    // Skip header and size info.
    while (fin.good() && std::getline(fin, line)) {
      if (line.length() == 0)
        continue;
      if (line[0] == '%')
        continue;
    }

    size_t k = 0;
    I i,j;
    T val;
    std::istringstream iss;
    if (mm_info.is_sparse) {

      while (fin.good() && std::getline(fin, line)) {
        iss.str(line);

        if (!(iss >> i) || (!(iss >> j)) || (!(iss >> val))) {
          fin.close();
          m.clear();
          return impl::errormsg
            (std::string("load_matrix_market failed to read line: ") + line,
             filepath);
        }
        --i; --j;     // 1-based indices
        if (i >= mm_info.num_rows || j >= mm_info.num_cols) {
          fin.close();
          m.clear();
          return impl::errormsg
            (std::string("load_matrix_market found invalid indices: ") + line,
             filepath);
        }
        m.row_index(k) = i;
        m.col_index(k) = j;
        m.value(k) = val;
        ++k;
      }

    } else { // dense matrix

      // 1 value per line
      // Listed by column

      i = 0; j = 0;
      while (fin.good() && std::getline(fin, line)) {
        iss.str(line);

        if (!(iss >> val)) {
          fin.close();
          m.clear();
          return impl::errormsg
            (std::string("load_matrix_market failed to read line: ") + line,
             filepath);
        }
        m(i,j) = val;
        ++i;
        if (i >= mm_info.num_rows) {
          i = 0;
          ++j;
          if (j >= mm_info.num_cols)
            break;
        }
      }
      if (i != mm_info.num_rows || j != mm_info.num_cols) {
        fin.close();
        m.clear();
        return impl::errormsg
          (std::string("load_matrix_market ended with (i,j)=(")
           + to_string(i) + ", " + to_string(j) + "), but (M,N)=("
           + to_string(mm_info.num_rows) + ", " + to_string(mm_info.num_cols)
           + ")", filepath);
      }
    }

    fin.close();
    return 0;

  } // load_matrix_market(coo_matrix<T,I>& m, const std::string& filepath)


  /**
   * Saves a matrix in Matrix Market array (dense) format.
   * @return  0: success, 1: failure
   */
  template <typename T>
  int save_matrix_market(const arma::Mat<T>& m, const std::string& filepath) {

    using namespace mm;
    MM_typecode matcode;                        

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_real(&matcode);

    std::ofstream fout(filepath.c_str());
    if (!fout.good())
      return impl::errormsg("save_matrix_market could not open file.",
                            filepath);

    mm_write_banner(fout, matcode); 
    mm_write_mtx_array_size(fout, m.n_rows, m.n_cols);

    for (size_t j = 0; j < m.n_cols; ++j) {
      for (size_t i = 0; i < m.n_rows; ++i)
        fprintf(fout, "%.10g\n", m(i,j));
    }

    fout.close();
    return 0;

  } // save_matrix_market(const arma::Mat<T>& m, const std::string& filepath)


  /**
   * Saves a matrix in Matrix Market sparse format.
   * @return  0: success, 1: failure
   */
  template <typename T, typename I>
  int save_matrix_market(const csc_matrix<T,I>& m, const std::string& filepath){
    // 1-based indices

    using namespace mm;
    MM_typecode matcode;                        

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    FILE *f = NULL;
    if ((f = fopen(filepath.c_str(), "w")) == NULL)
      return impl::errormsg("save_matrix_market could not open file.",
                            filepath);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, (int)m.n_rows, (int)m.n_cols,
                          (int)m.num_non_zeros());

    for (size_t j = 0; j < m.n_cols; ++j) {
      sparse_vector_view<T,I> col_j(m.col(j));
      for (size_t k = 0; k < col_j.num_non_zeros(); ++k)
        fprintf(f, "%d %d %.10g\n",
                (int)col_j.index(k) + 1, (int)j + 1, col_j.value(k));
    }

    fclose(f);
    f = NULL;
    return 0;

  } // save_matrix_market(const csc_matrix<T,I>& m, const std::string& filepath)

} // namespace sill

#endif // #ifndef SILL_MATRIX_MARKET_HPP
