#ifndef PRL_LINEAR_ALGEBRA_PARSER_HPP
#define PRL_LINEAR_ALGEBRA_PARSER_HPP

#include <algorithm>

#include <prl/math/linear_algebra.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Class for parsing dense/sparse vector/matrix formats.
   */
  class linear_algebra_parser {

    // Public types
    //==========================================================================
  public:

    /**
     * Numeric types codes.
     * These correspond to those used in my NumericType2PRLCode function in
     * Matlab.
     */
    enum matlab_type_code {
      MATLAB_UINT,
      MATLAB_UINT8,
      MATLAB_UINT16,
      MATLAB_UINT32,
      MATLAB_UINT64,
      MATLAB_UCHAR,
      MATLAB_UNSIGNED_CHAR,
      MATLAB_USHORT,
      MATLAB_ULONG,
      MATLAB_INT,
      MATLAB_INT8,
      MATLAB_INT16,
      MATLAB_INT32,
      MATLAB_INT64,
      MATLAB_INTEGER_1,
      MATLAB_INTEGER_2,
      MATLAB_INTEGER_3,
      MATLAB_INTEGER_4,
      MATLAB_SCHAR,
      MATLAB_SIGNED_CHAR,
      MATLAB_SHORT,
      MATLAB_LONG,
      MATLAB_SINGLE,
      MATLAB_DOUBLE,
      MATLAB_FLOAT,
      MATLAB_FLOAT32,
      MATLAB_FLOAT64,
      MATLAB_REAL_4,
      MATLAB_REAL_8,
      MATLAB_CHAR_1,
      MATLAB_CHAR
    }; // enum matlab_type_code

    template <typename T>
    struct coo_matrix_element {
      int row;
      int col;
      T val;
    }; // struct coo_matrix_element

    // Public methods
    //==========================================================================

    /**
     * Buffered vector read.
     *
     * @tparam VectorType   Type of vector y.
     * @tparam StorageType  Type which the data elements are stored as.
     *
     * @param f        File handle from which to read.
     * @param m        Number of elements in the vector.
     * @param y        (Return value) Pre-allocated vector into which to read
     *                 the values. The size is NOT checked!
     */
    template <typename VectorType, typename StorageType>
    static void read_binary_vec(FILE* f, size_t m, VectorType& y) {
      assert(f);
      const size_t BUFSIZE = std::min<size_t>(m, 500000);
      size_t toread = std::min<size_t>(m, BUFSIZE);
      size_t remain = m;

      size_t y_i(0);
      while (remain > 0) {
        StorageType * temp = new StorageType[remain];
        toread = (remain < BUFSIZE) ? remain : toread;
        size_t rc = fread(temp, sizeof(StorageType), toread, f);
        //assert(rc == toread);
        remain -= rc;
        for (size_t i(0); i < rc; ++i){
          y[y_i] = temp[i];
          ++y_i;
        }
        delete [] temp;
      }
    } // read_binary_vec()

    /**
     * Buffered vector read.
     *
     * @tparam VectorType   Type of vector y.
     *
     * @param f          File handle from which to read.
     * @param m          Number of elements in the vector.
     * @param y          (Return value) Pre-allocated vector into which to read
     *                   the values.
     * @param type_code  Specifies the storage type of elements in the file.
     */
    template <typename VectorType>
    static void read_binary_vec(FILE* f, size_t m, VectorType& y,
                                matlab_type_code type_code) {
      switch (type_code) {
      case MATLAB_UINT:
        read_binary_vec<VectorType, unsigned int>(f, m, y);
        break;
      case MATLAB_UINT8:
        read_binary_vec<VectorType, uint8_t>(f, m, y);
        break;
      case MATLAB_UINT16:
        read_binary_vec<VectorType, uint16_t>(f, m, y);
        break;
      case MATLAB_UINT32:
        read_binary_vec<VectorType, uint32_t>(f, m, y);
        break;
      case MATLAB_UINT64:
        read_binary_vec<VectorType, uint64_t>(f, m, y);
        break;
      case MATLAB_UCHAR:
      case MATLAB_UNSIGNED_CHAR:
        read_binary_vec<VectorType, unsigned char>(f, m, y);
        break;
      case MATLAB_USHORT:
        read_binary_vec<VectorType, unsigned short>(f, m, y);
        break;
      case MATLAB_ULONG:
        read_binary_vec<VectorType, unsigned long>(f, m, y);
        break;
      case MATLAB_INT:
        read_binary_vec<VectorType, int>(f, m, y);
        break;
      case MATLAB_INT8:
      case MATLAB_INTEGER_1:
        read_binary_vec<VectorType, int8_t>(f, m, y);
        break;
      case MATLAB_INT16:
      case MATLAB_INTEGER_2:
        read_binary_vec<VectorType, int16_t>(f, m, y);
        break;
      case MATLAB_INT32:
      case MATLAB_INTEGER_3:
        read_binary_vec<VectorType, int32_t>(f, m, y);
        break;
      case MATLAB_INT64:
      case MATLAB_INTEGER_4:
        read_binary_vec<VectorType, int64_t>(f, m, y);
        break;
      case MATLAB_SCHAR:
      case MATLAB_SIGNED_CHAR:
        read_binary_vec<VectorType, signed char>(f, m, y);
        break;
      case MATLAB_SHORT:
        read_binary_vec<VectorType, short>(f, m, y);
        break;
      case MATLAB_LONG:
        read_binary_vec<VectorType, long>(f, m, y);
        break;
      case MATLAB_SINGLE:
      case MATLAB_FLOAT:
      case MATLAB_FLOAT32:
      case MATLAB_REAL_4:
        read_binary_vec<VectorType, float>(f, m, y);
        break;
      case MATLAB_DOUBLE:
      case MATLAB_FLOAT64:
      case MATLAB_REAL_8:
        read_binary_vec<VectorType, double>(f, m, y);
        break;
      case MATLAB_CHAR_1:
      case MATLAB_CHAR:
        read_binary_vec<VectorType, char>(f, m, y);
        break;
      default:
        assert(false);
      }
    } // read_binary_vec() given matlab_type_code

    /**
     * Read a dense PRL matrix from a file.
     *
     * Format:
     *   m  (unsigned int)  (number of rows)
     *   n  (unsigned int)  (number of columns)
     *   value_type (unsigned int) (See matlab_type_codes.)
     *   column_1 (vector of value_type of length m)
     *   column_2 ...
     *
     * @tparam T   Type of elements.
     *
     * @param A         (Return value) Matrix read from file.
     * @param filepath  File location.
     */
    template <typename T>
    static void read_dense_matrix(matrix<T>& A, const std::string& filepath) {
      FILE* f = fopen(filepath.c_str(), "r");
      assert(f);

      unsigned int m(0);
      unsigned int n(0);
      unsigned int value_type_code(0);

      size_t rc = fread(&m, sizeof(unsigned int), 1, f);
      assert(rc == 1);
      rc = fread(&n, sizeof(unsigned int), 1, f);
      assert(rc == 1);
      rc = fread(&value_type_code, sizeof(unsigned int), 1, f);
      assert(rc == 1);

      assert(m > 0 && n > 0);
      A.resize(m, n);

      T* data_ptr = A._data();
      read_binary_vec(f, m * n, data_ptr, (matlab_type_code)value_type_code);

      fclose(f);
      f = NULL;
    } // read_dense_matrix()

    /**
     * Read a dense PRL vector from a file.
     *
     * Format:
     *   n  (unsigned int)   (number of elements)
     *   value_type (unsigned int) (See matlab_type_codes.)
     *   values (vector of value_type of length n)
     *
     * @tparam T   Type of elements.
     *
     * @param v         (Return value) Vector read from file.
     * @param filepath  File location.
     */
    template <typename T>
    static void read_dense_vector(vector<T>& v, const std::string& filepath) {
      FILE* f = fopen(filepath.c_str(), "r");
      if (!f) {
        throw std::invalid_argument
          ("linear_algebra_parser::read_dense_vector() could not open filepath: "
           + filepath);
      }

      unsigned int n(0);
      unsigned int value_type_code(0);

      size_t rc = fread(&n, sizeof(unsigned int), 1, f);
      assert(rc == 1);
      rc = fread(&value_type_code, sizeof(unsigned int), 1, f);
      assert(rc == 1);

      assert(n > 0);
      v.resize(n);

      T* data_ptr = v._data();
      read_binary_vec(f, n, data_ptr, (matlab_type_code)value_type_code);

      fclose(f);
      f = NULL;
    } // read_dense_vector()

  }; // class linear_algebra_parser

}; // namespace prl

#include <prl/macros_undef.hpp>

#endif // PRL_LINEAR_ALGEBRA_PARSER_HPP
