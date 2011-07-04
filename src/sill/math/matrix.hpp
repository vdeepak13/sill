#ifndef SILL_MATRIX_HPP
#define SILL_MATRIX_HPP

#warning "sill/math/matrix.hpp is deprecated"

#include <functional>
#include <string>

#include <itpp/base/vec.h>
#include <itpp/base/mat.h>
// we should also include the functions over vectors & matrices

#include <sill/global.hpp>
#include <sill/math/irange.hpp>
#include <sill/math/vector.hpp>
#include <sill/stl_concepts.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  // We reimplement some of the functions, to maintain a complete 
  // documentation

  /**
   * A dense matrix.
   *
   * This class extends its functionality of IT++ matrices to better
   * work with submatrices.  For all practical purposes, this class
   * behaves just like the original IT++ class (it can be used with
   * all the operators and functions in IT++), but it provides a
   * cleaner interface and serialization. For more information on
   * related functions, see
   * http://itpp.sourceforge.net/current/classitpp_1_1Mat.html
   *
   * Note that the non-trivial indexing functions (such as 
   * operator()(irange)) return a copy of the corresponding
   * elements. This convention differs from Matlab, where one can
   * write v([1 2 3]) = 3 (thus, in Matlab, indexing creates
   * a reference to a subvector). To assign or manipulate subvectors,
   * use the functions set_subvector, add_subvector, etc.
   *
   * All indexing is 0-based. The data is stored in a column-major
   * order (the Matlab / Fortran convention).
   *
   * \ingroup math_linalg
   */
  template <class T>
  class matrix : public itpp::Mat<T> {

  #ifdef _MSC_VER  
    // in VC++, itpp::Mat::T() shadows the template argument T
    typedef value_type T;
  #endif
    
    // Public type declarations
    //==========================================================================
  public:
    //! The type of values stored in this matrix
    typedef T value_type;

    //! The base type
    typedef itpp::Mat<T> base;

    // Container interface
    typedef std::ptrdiff_t  difference_type;
    typedef size_t          size_type;
    typedef T&              reference;
    typedef T*              pointer;
    typedef T*              iterator;
    typedef const T&        const_reference;
    typedef const T*        const_pointer;
    typedef const T*        const_iterator;

    // Constructors
    //==========================================================================
  public:
    //! Default constructor
    matrix() { }

    //! Creates a matrix with of size (nrows, ncols)
    //! Note: the matrix is not necessarily zero-initialized
    matrix(size_t nrows, size_t ncols) : base(nrows, ncols) {
      assert(nrows * ncols <= (size_t)(std::numeric_limits<int>::max()));
    }

    //! Creates a matrix filled with the given element
    matrix(size_t nrows, size_t ncols, T value) : base(nrows, ncols) {
      assert(nrows * ncols <= (size_t)(std::numeric_limits<int>::max()));
      std::fill_n(base::data, nrows*ncols, value);
    }

    //! Creates a vector from a C-array (the array contents are copied).
    matrix(const T* array, size_t nrows, size_t ncols, bool row_major = true) 
      : base(array, nrows, ncols, row_major) {
      assert(nrows * ncols <= (size_t)(std::numeric_limits<int>::max()));
    }

    //! Conversion from the base class
    matrix(const base& mat) : base(mat) { }

    //! Creates a matrix that is equivalent to a column vector
    explicit matrix(const itpp::Vec<T>& vec) : base(vec) { }

    //! Conversion from human-readable representation
    matrix(const std::string& str) : base(str) { }
    
    //! Conversion from human-readable representation
    matrix(const char* str) : base(str) { }

    // Range interface
    //==========================================================================
    //! Returns a pointer to the first element
    const T* begin() const {
      return base::_data();
    }

    //! Returns a pointer to the first element
    T* begin() {
      return base::_data();
    }

    //! Returns a pointer past the last element
    const T* end() const {
      return begin() + size();
    }

    //! Returns a pointer past the last element
    T* end() {
      return begin() + size();
    }

    // Accessors
    //==========================================================================

    using base::cols;
    using base::rows;

    //! The number of rows
    size_t n_rows const {
      return base::no_rows;
    }

    //! The number of columns
    size_t n_cols const {
      return base::no_cols;
    }

    //! The number of elements in the matrix
    size_t size() const {
      return base::datasize;
    }

    //! Returns true if the matrix has no elements
    bool empty() const {
      return base::datasize == 0;
    }
    
    //! Returns the element \c (i, j)
    const T& operator()(size_t i, size_t j) const{
      return base::operator()(i, j);
    }

    //! Returns the element \c (i,j)
    T& operator()(size_t i, size_t j) {
      return base::operator()(i, j);
    }

    //! Returns an element using linear addressing
    const T& operator()(size_t index) const {
      return base::operator()(index);
    }

    //! Returns an element using linear addressing
    T& operator()(size_t index) {
      return base::operator()(index);
    }

    //! Returns a continuous block of the matrix
    const matrix operator()(irange i, irange j) const {
      if (i.empty() || j.empty())
        return itpp::Mat<T>(i.size(), j.size());
      else
        return base::get(i.start(), i.end(), j.start(), j.end());
    }

    //! Returns a submatrix over rows \c i and columns \c j
    //! \todo This function is not very efficient yet
    const matrix operator()(const itpp::uvec& i, 
                            const itpp::uvec& j) const {
      itpp::Mat<T> a(i.size(), j.size());
      for(int k = 0; k < i.size(); k++)
        for(int l = 0; l < j.size(); l++)
          a.set(k, l, operator()(i(k), j(l)));
      return a;
    }
    
    //! Returns the row \c i
    const itpp::Vec<T> row(size_t i) const {
      return base::get_row(i); 
    }

    //! Returns the rows in a continuous range
    const itpp::Mat<T> rows(irange i) const {
      if (i.empty())
        return itpp::Mat<T>(0, n_cols);
      else
        return base::get_rows(i.start(), i.end());
    }

    //! Returns the rows with indices \c i
    const itpp::Mat<T> rows(const itpp::uvec& i) const {
      return base::get_rows(i);
    }

    //! Returns the column \c i
    const vector<T> column(size_t i) const {
      return base::get_col(i);
    }

    //! Returns the columns in a continuous range
    const itpp::Mat<T> columns(irange i) const {
      if (i.empty())
        return itpp::Mat<T>(n_rows, 0);
      else
        return base::get_cols(i.start(), i.end());
    }

    //! Returns the columns with indices \c i
    const itpp::Mat<T> columns(const itpp::uvec& i) const {
      return base::get_cols(i);
    }

    //! Returns the transpose of the matrix
    matrix transpose() const {
      if (size() == 0)
        return matrix();
//        return matrix(n_cols, n_rows);
      else
        return base::transpose();
    }

    //! Returns the conjugate transpose of the matrix
    const itpp::Mat<T> hermitian_transpose() const {
      return base::hermitian();
    }

    // Comparison operators
    //==========================================================================
    //! Returns true if the matrices have the same dimensions and values
    bool operator==(const itpp::Mat<T>& m) const {
      return base::operator==(m);
    }
    
    //! Returns true if the matrices have different dimensions or values.
    bool operator!=(const itpp::Mat<T>& m) const {
      return base::operator!=(m);
    }

    // Updates
    //==========================================================================
    //! Sets all elements of the matrix to \c v
    matrix& operator=(T v) {
      base::operator=(v); return *this;
    }

    //! Assigns a matrix
    matrix& operator=(const itpp::Mat<T>& m) {
      base::operator=(m); return *this;
    }

    //! Assigns the matrix to the matrix represented by the string
    matrix& operator=(const char* str) {
      base::operator=(str); return *this;
    }

    //! Changes the dimensions of the matrix. 
    //! \param copy if true, retains the original data
    void resize(size_t nrows, size_t ncols, bool copy = false) { 
      assert(nrows * ncols <= (size_t)(std::numeric_limits<int>::max()));
      base::set_size(nrows, ncols, copy);
    }
    
    //! Sets all entries of to zero
    void clear() {
      base::zeros();
    }

    //! Sets all entries of to zero
    void zeros() {
      base::zeros();
    }

    //! Sets all entries to zero using memset.
    //! (Joseph: I kept this separate, but feel free to replace zeros()
    //!  with this.)
    void zeros_memset() {
      memset(base::_data(), 0, sizeof(T) * base::_datasize());
    }

    //! Sets all entries of to one
    void ones() {
      base::ones();
    }

    //! Sets a block of the matrix
    void set_submatrix(irange i, irange j, const itpp::Mat<T>& m) {
      if (i.size() != (size_t)(m.rows()) || j.size() != (size_t)(m.cols())) {
        throw std::invalid_argument
          (std::string("matrix<T>::set_submatrix(i,j,m) given i,j not") +
           " matching dimensionality of m.");
      }
      if (!i.empty() && !j.empty())
        base::set_submatrix(i.start(), j.start(), m);
//        base::set_submatrix(i.start(), i.end(), j.start(), j.end(), m);
    }

    //! Sets a block of the matrix to a constant
    void set_submatrix(irange i, irange j, T value) {
      if (!i.empty() && !j.empty())
        base::set_submatrix(i.start(), i.end, j.start(), j.end(), value);
    }

    //! Sets a submatrix with rows \c i and columns \c j
    //! \todo This function is not very efficient at the moment
    void set_submatrix(const itpp::uvec& i, const itpp::uvec& j, 
                       const itpp::Mat<T>& m) {
      assert(i.size() == m.rows());
      assert(j.size() == m.cols());
      for(int k = 0; k < i.size(); k++)
        for(int l = 0; l < j.size(); l++)
          base::set(i(k), j(l), m(k,l));
    }

    //! Sets a submatrix with rows \c i and columns \c j to a constant
    void set_submatrix(const itpp::uvec& i, const itpp::uvec& j, T value) {
      for(size_t k = 0; k < i.size(); k++)
        for(size_t l = 0; l < j.size(); l++)
          base::set(i(k), j(l), value);
    }

    //! Sets row \c i to a vector
    void set_row(size_t i, const itpp::Vec<T>& v) {
      base::set_row(i, v);
    }

    //! Sets a continuous range of rows, starting from \c i
    void set_rows(size_t i, const itpp::Mat<T>& m) {
      base::set_rows(i, m);
    }

    //! Sets column \c j to a vector
    void set_column(size_t j, const itpp::Vec<T>& v) {
      base::set_col(j, v);
    }

    //! Sets a continous range of columns, starting from \c j
    void set_column(size_t j, const itpp::Mat<T>& m) {
      base::set_cols(j, m);
    }
    
    //! Copy row \c from to row \c to.
    //! (note that the order of arguments differs from IT++).
    void copy_row(size_t from, size_t to) {
      base::copy_row(to, from);
    }
    
    //! Copy column \c from to column \c to.
    //! (note that the order of arguments differs from IT++).
    void copy_column(size_t from, size_t to) {
      base::copy_col(to, from);
    }

    //! Swap two rows
    void swap_rows(size_t i1, size_t i2) {
      base::swap_rows(i1, i2);
    }

    //! Swap two columns
    void swap_columns(size_t j1, size_t j2) {
      base::swap_cols(j1, j2);
    }

    //! Deletes a row
    void delete_row(size_t i) {
      base::del_row(i);
    }

    //! Deletes a continuous range of row
    void delete_rows(irange i) {
      if (!i.empty())
        base::del_rows(i.start(), i.end());
    }

    //! Deletes a column
    void delete_column(size_t j) {
      base::del_col(j);
    }

    //! Deletes a continuous range of columns
    void delete_columns(irange j) {
      if (!j.empty())
        base::del_cols(j.start(), j.end());
    }

    //! Inserts a new row at index i; the matrix can be empty.
    void insert_row(size_t i, const itpp::Vec<T>& v) {
      base::ins_row(i, v);
    }

    //! Inserts a new column at index j; the matrix can be empty.
    void insert_column(size_t j, const itpp::Vec<T>& v) {
      base::ins_col(j, v);
    }
    
    //! Appends a vector to the bottom of the matrix; the matrix can be empty.
    void append_row(const itpp::Vec<T>& v) {
      base::append_row(v);
    }

    //! Appends a vector to the right of the matrix; the matrix can be empty.
    void append_column(const itpp::Vec<T>& v) {
      base::append_column(v);
    }

    /**
     * Updates a submatrix with a binary function
     * @tparam F a type that satisfies the BinaryFunction concept
     * \todo this function is not very efficient at the moment.
     */
    template <typename F>
    void update_submatrix(irange i, irange j, const itpp::Mat<T>& a, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(i.size() == a.rows());
      assert(j.size() == a.cols());
      for(size_t r = 0; r < i.size(); r++) {
        for(size_t c = 0; c < j.size(); c++) {
          size_t ir = i(r), jc = j(c);
          base::set(ir, jc, f(operator()(ir, jc), a(r, c)));
        }
      }
    }

    /**
     * Updates a submatrix with a binary function
     * @tparam F a type that satisfies the BinaryFunction concept
     * \todo this function is not very efficient at the moment.
     */
    template <typename F>
    void update_submatrix(const itpp::uvec& i, const itpp::uvec& j,
                          const itpp::Mat<T>& a, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(i.size() == a.rows());
      assert(j.size() == a.cols());
      // use int here to avoid signed-unsigned comparison warnings
      for(int r = 0; r < i.size(); r++) {
        for(int c = 0; c < j.size(); c++) {
          size_t ir = i(r), jc = j(c);
          base::set(ir, jc, f(operator()(ir, jc), a(r, c)));
        }
      }
    }

    /**
     * Updates a submatrix with a binary function
     * @tparam F a type that satisfies the BinaryFunction concept
     * \todo this function is not very efficient at the moment.
     */
    template <typename F>
    void update_submatrix(irange i, const itpp::uvec& j,
                          const itpp::Mat<T>& a, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(i.size() == (size_t)(a.rows()));
      assert(j.size() == a.cols());
      for(size_t r = 0; r < i.size(); r++) {
        for(size_t c = 0; c < (size_t)(j.size()); c++) {
          size_t ir = i(r), jc = j(c);
          base::set(ir, jc, f(operator()(ir, jc), a(r, c)));
        }
      }
    }

    //! Updates row i with a binary function
    //! @tparam F a type that satisfies the BinaryFunction concept
    template <typename F>
    void update_row(size_t i, const itpp::Vec<T>& v, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(this->cols() == v.size());
      for(size_t j = 0; j < n_cols; j++) 
        base::set(i, j, f(operator()(i, j), v[j]));
    }

    //! Updates a row with a binary function
    //! @tparam F a type that satisfies the BinaryFunction concept
    template <typename F>
    void update_row(size_t i, T v, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      for(size_t j = 0; j < n_cols; j++)
        base::set(i, j, f(operator()(i, j), v));
    }

    //! Updates row i with a binary function, scaling the vector v
    //! by alpha:
    //!   new_row(j) <-- f(old_row(j), alpha * v(j))
    //! @tparam F a type that satisfies the BinaryFunction concept
    template <typename F>
    void scaled_update_row(size_t i, const itpp::Vec<T>& v, F f, T alpha) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(this->cols() == v.size());
      for(size_t j = 0; j < n_cols; j++) 
        base::set(i, j, f(operator()(i, j), alpha * v[j]));
    }

    //! Updates a column with a binary function
    //! @tparam F a type that satisfies the BinaryFunction concept
    template <typename F>
    void update_column(size_t j, const itpp::Vec<T>& v, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      assert(this->rows() == v.size());
      for(size_t i = 0; i < n_rows; i++) 
        base::set(i, j, f(operator()(i, j), v[i]));
    }

    //! Updates a column with a binary function
    //! @tparam F a type that satisfies the BinaryFunction concept
    template <typename F>
    void update_column(size_t j, T v, F f) {
      concept_assert((BinaryFunction<F, T, T, T>));
      for(size_t i = 0; i < n_rows; i++)
        base::set(i, j, f(operator()(i, j), v));
    }

    // Arithmetic operations
    //==========================================================================
    //! Matrix addition
    matrix& operator+=(const itpp::Mat<T>& m) {
      base::operator+=(m); return *this;
    }

    //! Adds a scalar to the matrix
    matrix& operator+=(T value) {
      base::operator+=(value); return *this;
    }

    //! Matrix subtraction
    matrix& operator-=(const itpp::Mat<T>& m) {
      base::operator-=(m); return *this;
    }

    //! Subtracts a scalar from the matrix
    matrix& operator-=(T value) {
      base::operator-=(value); return *this;
    }

    //! Matrix-matrix multiplication
    matrix& operator*=(const matrix& m) {
      if (size() == 0 || m.size() == 0) {
        if (n_cols != m.n_rows)
          throw std::invalid_argument
            ("matrix<T>::operator*=(m) given m with non-matching dimensions");
        this->resize(n_rows, m.n_cols);
        this->zeros();
      } else {
        base::operator*=(m);
      }
      return *this;
    }

    //! Multiplies the matrix by a scalar
    matrix& operator*=(T value) {
      base::operator*=(value); return *this;
    }

    //! Element-wise division by a matrix
    matrix& operator/=(const itpp::Mat<T>& m) {
      base::operator/=(m); return *this;
    }

    //! Divides the matrix by a scalar
    matrix& operator/=(T value) {
      base::operator/=(value); return *this;
    }

    //! Adds to a submatrix (i, j)
    void add_submatrix(irange i, irange j, const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::plus<T>());
    }

    //! Adds to a submatrix (i, j)
    void add_submatrix(const itpp::uvec& i, const itpp::uvec& j, 
                       const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::plus<T>());
    }

    //! Adds to a submatrix (i, j)
    void add_submatrix(irange i, const itpp::uvec& j, const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::plus<T>());
    }

    //! Subtracts from a submatrix (i, j)
    void subtract_submatrix(irange i, irange j, const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::minus<T>());
    }

    //! Subtracts from a submatrix (i, j)
    void subtract_submatrix(const itpp::uvec& i, const itpp::uvec& j, 
                            const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::minus<T>());
    }

    //! Subtracts from a submatrix (i, j)
    void subtract_submatrix(irange i, const itpp::uvec& j,
                            const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::minus<T>());
    }

    //! Divides a submatrix (i, j) element-wise
    void divide_submatrix(irange i, irange j, const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::divides<T>());
    }

    //! Divides a submatrix (i, j) element-wise
    void divide_submatrix(const itpp::uvec& i, const itpp::uvec& j, 
                          const itpp::Mat<T>& m) {
      update_submatrix(i, j, m, std::divides<T>());
    }

    //! Adds a vector to row i
    void add_row(size_t i, const itpp::Vec<T>& v) {
      update_row(i, v, std::plus<T>());
    }

    //! Adds a constant to a row
    void add_row(size_t i, T v) {
      update_row(i, v, std::plus<T>());
    }

    //! Adds a scaled vector alpha * v to row i
    void add_scaled_row(size_t i, const itpp::Vec<T>& v, T alpha) {
      scaled_update_row(i, v, std::plus<T>(), alpha);
    }

    //! Subtracts a vector from a row
    void subtract_row(size_t i, const itpp::Vec<T>& v) {
      update_row(i, v, std::minus<T>());
    }

    //! Subtracts a constant from a row
    void subtract_row(size_t i, T v) {
      update_row(i, v, std::minus<T>());
    }

    //! Multiplies a row by a vector element-wise
    void multiply_row(size_t i, const itpp::Vec<T>& v) {
      update_row(i, v, std::multiplies<T>());
    }
    
    //! Multiplies a row by a constant
    void multiply_row(size_t i, T v) {
      update_row(i, v, std::multiplies<T>());
    }
    
    //! Divides a row by a vector element-wise
    void divide_row(size_t i, const itpp::Vec<T>& v) {
      update_row(i, v, std::divides<T>());
    }
    
    //! Divides a row by a constant
    void divide_row(size_t i, T v) {
      update_row(i, v, std::divides<T>());
    }

    //! Adds a vector to a column
    void add_column(size_t j, const itpp::Vec<T>& v) {
      update_column(j, v, std::plus<T>());
    }

    //! Adds a constant to a column
    void add_column(size_t j, T v) {
      update_column(j, v, std::plus<T>());
    }

    //! Subtracts a vector from a column
    void subtract_column(size_t j, const itpp::Vec<T>& v) {
      update_column(j, v, std::minus<T>());
    }

    //! Subtracts a constant from a column
    void subtract_column(size_t j, T v) {
      update_column(j, v, std::minus<T>());
    }

    //! Multiplies a column by a vector element-wise
    void multiply_column(size_t j, const itpp::Vec<T>& v) {
      update_column(j, v, std::multiplies<T>());
    }
    
    //! Multiplies a column by a constant 
    void multiply_column(size_t j, T v) {
      update_column(j, v, std::multiplies<T>());
    }
    
    //! Divides a column by a vector element-wise
    void divide_column(size_t j, const itpp::Vec<T>& v) {
      update_column(j, v, std::divides<T>());
    }
    
    //! Divides a column by a constant
    void divide_column(size_t j, T v) {
      update_column(j, v, std::divides<T>());
    }

    // Methods to support optimization routines
    //==========================================================================

    //! Inner product
    T inner_prod(const itpp::Mat<T>& x) const {
      return elem_mult_sum(*this, x);
    }

    //! Returns the L1 norm.
    double L1norm() const {
      return sumsum(abs(*this));
    }

    //! Returns the L2 norm.
    double L2norm() const {
      return sqrt(inner_prod(*this));
    }

    // Serialization
    //==========================================================================
    void save(oarchive& ar) const {
      ar << n_rows;
      ar << n_cols;
      for(size_t i = 0; i < size(); i++)
        ar << operator()(i);
    }
  
    void load(iarchive& ar) {
      size_t m, n;
      ar >> m >> n;
      resize(m, n);
      for(size_t i = 0; i < size(); i++)
        ar >> operator()(i);
    }

    /**
     * Print in human-readable format.
     * Separate each element with elem_delimiter (default = space),
     * and separate each row with row_delimiter (default = newline).
     */
    void print(std::ostream& out, const std::string& elem_delimiter = " ",
               const std::string& row_delimiter = "\n") const {
      for (size_t i(0); i < n_rows; ++i) {
        for (size_t j(0); j < n_cols; ++j) {
          out << operator()(i,j);
          if (j + 1 == n_cols)
            out << row_delimiter;
          else
            out << elem_delimiter;
        }
      }
    }

  }; // class matrix

  // Operator overrides to deal with IT++ bugs for empty matrices
  //============================================================================

  template <typename T>
  matrix<T> operator+(const matrix<T>& m1, const matrix<T>& m2) {
    matrix<T> m(m1);
    m += m2;
    return m;
  }

  template <typename T>
  matrix<T> operator-(const matrix<T>& m1, const matrix<T>& m2) {
    matrix<T> m(m1);
    m -= m2;
    return m;
  }

  template <typename T>
  matrix<T> operator*(const matrix<T>& m1, const matrix<T>& m2) {
    matrix<T> m(m1);
    m *= m2;
    return m;
  }

  // Standardized free functions
  //============================================================================
  
  //! \relates matrix
  template <typename T>
  itpp::Mat<T> trans(const itpp::Mat<T>& a) {
    return a.transpose();
  }

  //! \relates matrix
  template <typename T>
  itpp::Mat<T> herm(const itpp::Mat<T>& a) {
    return a.hermitian_transpose();
  }

  //! Matrix dot product (treating the matrix as a vector)
  //! \relates matrix
  template <typename T>
  T elem_mult_sum(const matrix<T>& m1, const matrix<T>& m2) {
    return itpp::elem_mult_sum(m1,m2);
  }

  // Type definitions
  //============================================================================

  //! A real-valued matrix
  //! \relates matrix
  typedef matrix<double> mat;
  
  //! A complex-valued matrix
  //! \relates matrix
  typedef matrix<std::complex<double> > cmat;

  //! An integer-valued matrix
  //! \relates matrix
  typedef matrix<int> imat;

  //! A short integer-valued matrix
  //! \relates matrix
  typedef matrix<short> smat;

  //! A matrix with entries over GF(2) 
  //! \relates matrix
  typedef matrix<itpp::bin> bmat;

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
