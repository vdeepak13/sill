#ifndef SILL_ARMADILLO_HPP
#define SILL_ARMADILLO_HPP

/**
 * \file armadillo.hpp  Includes dense linear algebra headers.
 *
 * @todo Change this file's name to dense_linear_algebra.hpp?
 */

#include <armadillo>

#include <sill/base/stl_util.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/serialization/iterator.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  template <typename Ref> class forward_range;

  // Bring Armadillo's types and functions into the sill namespace
  //============================================================================

  // matrix and vector types
  using arma::imat;
  using arma::umat;
  using arma::fmat;
  using arma::mat;
  using arma::cx_fmat;
  using arma::cx_mat;
  
  using arma::ivec;
  using arma::uvec;
  using arma::fvec;
  using arma::vec;
  using arma::cx_fvec;
  using arma::cx_vec;

  using arma::icolvec;
  using arma::ucolvec;
  using arma::fcolvec;
  using arma::colvec;
  using arma::cx_fcolvec;
  using arma::cx_colvec;

  using arma::irowvec;
  using arma::urowvec;
  using arma::frowvec;
  using arma::rowvec;
  using arma::cx_frowvec;
  using arma::cx_rowvec;

  // span of indices
  using arma::span;

  // generated vectors and matrices
  using arma::eye;
  using arma::linspace;
  using arma::randu;
  using arma::randn;
  using arma::zeros;
  using arma::ones;

  // functions of vectors and matrices
  using arma::dot;
  using arma::norm;

  // scalar (remove this eventually)
  using arma::as_scalar;

  //============================================================================

  /**
   * Dense linear algebra specification.
   *
   * This type of struct can be passed to methods as a template parameters
   * to specify what vector/matrix classes should be used.
   *
   * STANDARD: Classes which take a linear algebra specifier as a template
   *           parameter (or have one hard-coded) should typedef the specifier
   *           as "la_type" as a standard name for other classes to use.
   */
  template <typename T = double, typename SizeType = arma::u32>
  struct dense_linear_algebra {

    typedef arma::Col<T>  vector_type;
    typedef arma::Mat<T>  matrix_type;
    typedef typename vector_type::value_type value_type;
    typedef typename vector_type::size_type  size_type;

    typedef arma::Col<T>  dense_vector_type;
    typedef arma::Mat<T>  dense_matrix_type;

    typedef uvec  index_vector_type;

  };

  // (temporary) Functions which would be nice to have in Armadillo
  //============================================================================

  //! Resize a vector.
  template <typename T>
  arma::Col<T> resize(const arma::Col<T>& a, size_t n) {
    arma::Col<T> b(zeros(n));
    size_t m = std::min<size_t>(a.size(), n);
    for (size_t i = 0; i < m; ++i)
      b[i] = a[i];
    return b;
  }

  //! Simpler log_det function for matrices with positive determinants.
  template <typename MatType>
  typename MatType::value_type log_det(const MatType& m) {
    typename MatType::value_type val;
    typename MatType::value_type s;
    arma::log_det(val, s, m);
    assert(s > 0);
    return val;
  }

  template <typename T>
  arma::Col<T> concat(const arma::Col<T>& v1, const arma::Col<T>& v2) {
    arma::Col<T> result(v1.size() + v2.size());
    if (!v1.is_empty())
      result.subvec(span(0,v1.size()-1)) = v1;
    if (!v2.is_empty())
      result.subvec(span(v1.size(),result.size()-1)) = v2;
    return result;
  }

  template <typename T>
  arma::Col<T> concat(const forward_range<const arma::Col<T>&> vectors) {
    // compute the size of the resulting vector
    size_t n = 0;
    foreach(const arma::Col<T>& v, vectors) n += v.size();
    arma::Col<T> result(n);

    // assign the vectors to the right indices
    n = 0;
    foreach(const arma::Col<T>& v, vectors) {
      if (!v.is_empty()) {
        result.subvec(span(n, n + v.size() - 1)) = v;
      }
      n += v.size();
    }
    return result;
  }

  template <typename T>
  arma::Col<T> concat(const std::vector<arma::Col<T> >& vectors) {
    return concat(forward_range<const arma::Col<T>&>(vectors));
  }

  template <typename T>
  arma::Mat<T>
  concat_horizontal(const arma::Mat<T>& left, const arma::Mat<T>& right) {
    assert(left.n_rows == right.n_rows);
    arma::Mat<T> out(left.n_rows, left.n_cols + right.n_cols);
    if (left.n_cols != 0)
      out.submat(span::all, span(0,left.n_cols-1)) = left;
    if (right.n_cols != 0)
      out.submat(span::all, span(left.n_cols, out.n_cols-1)) = right;
    return out;
  }

  /**
   * Compare two matrices.
   * @return true iff exactly equal
   */
  template <typename T>
  bool equal(const arma::Mat<T>& a, const arma::Mat<T>& b) {
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols)
      return false;
    const T* a_it = a.begin();
    const T* a_end = a.end();
    const T* b_it = b.begin();
    while (a_it != a_end) {
      if (*a_it != *b_it)
        return false;
      ++a_it;
      ++b_it;
    }
    return true;
  }

  /**
   * Compare two matrices.
   * @todo If this is as fast as the above implementation for Mat,
   *       then get rid of the above one.
   * @return true iff exactly equal
   */
  template <typename MatrixType>
  bool equal(const MatrixType& a, const MatrixType& b) {
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols)
      return false;
    return (accu(a == b) == a.n_elem);
  }

  /**
   * Outer product free function.
   * @todo (Joseph B.) I added this to maintain compatability with my sparse
   *       linear algebra code.  This could be removed in the future once the
   *       sparse LA code distinguishes between column/row vectors.
   */
  template <typename T>
  arma::Mat<T> outer_product(const arma::Col<T>& a, const arma::Col<T>& b) {
    return a * trans(b);
  }

  //! Return a matrix with the selected columns from A.
  template <typename T>
  arma::Mat<T> columns(const arma::Mat<T>& A, const uvec& col_indices) {
    arma::Mat<T> m(A.n_rows, col_indices.size());
    for (size_t i = 0; i < col_indices.size(); ++i) {
      assert(col_indices[i] < A.n_cols);
      m.col(i) = A.col(col_indices[i]);
    }
    return m;
  }

  //! Set a submatrix: A(rows,cols) = B.
  template <typename T>
  void set_submatrix(arma::Mat<T>& A, const span& rows, const uvec& cols,
                     const arma::Mat<T>& B) {
    assert((rows.b+1 - rows.a) == B.n_rows && cols.size() == B.n_cols);
    for (size_t j = 0; j < cols.size(); ++j) {
      assert(cols[j] < A.n_cols);
      A(rows, cols[j]) = B.col(j);
    }
  }

  //! Set a submatrix: A(rows,cols) = B.
  template <typename T>
  void set_submatrix(arma::Mat<T>& A, const uvec& rows, const span& cols,
                     const arma::Mat<T>& B) {
    assert(rows.size() == B.n_rows && (cols.b+1 - cols.a) == B.n_cols);
    assert(cols.b < A.n_cols);
    for (size_t i = 0; i < rows.size(); ++i) {
      A(rows[i], cols) = B.row(i);
    }
  }

  //! Add a submatrix: A(rows,cols) += B.
  template <typename T>
  void add_submatrix(arma::Mat<T>& A, const span& rows, const uvec& cols,
                     const arma::Mat<T>& B) {
    assert((rows.b+1 - rows.a) == B.n_rows && cols.size() == B.n_cols);
    for (size_t j = 0; j < cols.size(); ++j) {
      assert(cols[j] < A.n_cols);
      A(rows, cols[j]) += B.col(j);
    }
  }

  //! Add a submatrix: A(rows,cols) += B.
  template <typename T>
  void add_submatrix(arma::Mat<T>& A, const uvec& rows, const span& cols,
                     const arma::Mat<T>& B) {
    assert(rows.size() == B.n_rows && (cols.b+1 - cols.a) == B.n_cols);
    assert(cols.b < A.n_cols);
    for (size_t i = 0; i < rows.size(); ++i) {
      A(rows[i], cols) += B.row(i);
    }
  }

  //! Subtract a submatrix: A(rows,cols) += B.
  template <typename T>
  void subtract_submatrix(arma::Mat<T>& A, const span& rows, const uvec& cols,
                     const arma::Mat<T>& B) {
    assert((rows.b+1 - rows.a) == B.n_rows && cols.size() == B.n_cols);
    for (size_t j = 0; j < cols.size(); ++j) {
      assert(cols[j] < A.n_cols);
      A(rows, cols[j]) -= B.col(j);
    }
  }

  //! Subtract a submatrix: A(rows,cols) += B.
  template <typename T>
  void subtract_submatrix(arma::Mat<T>& A, const uvec& rows, const span& cols,
                     const arma::Mat<T>& B) {
    assert(rows.size() == B.n_rows && (cols.b+1 - cols.a) == B.n_cols);
    assert(cols.b < A.n_cols);
    for (size_t i = 0; i < rows.size(); ++i) {
      A(rows[i], cols) -= B.row(i);
    }
  }

  /*
  //! Set the submatrix A(rows,cols) = B.
  template <typename T>
  void set_submatrix(arma::Mat<T>& A, const uvec& rows, const uvec& cols,
                     const arma::Mat<T>& B) {
    assert(rows.size() == B.n_rows && cols.size() == B.n_cols);
    for (size_t j = 0; j < cols.size(); ++j) {
      assert(cols[j] < A.n_cols);
      A.col(cols[j]).elem(rows) = B.col(j);
    }
  }
  */

  //! Constructs a 1x1 matrix [a].
  template <typename T>
  arma::Mat<T> mat_1x1(T a) {
    arma::Mat<T> m(1,1);
    m(0,0) = a;
    return m;
  }

  //! Constructs a 2x2 matrix [a b; c d].
  template <typename T>
  arma::Mat<T> mat_2x2(T a, T b, T c, T d) {
    arma::Mat<T> m(2,2);
    m(0,0) = a;
    m(0,1) = b;
    m(1,0) = c;
    m(1,1) = d;
    return m;
  }

  //! Constructs a 3x3 matrix [a b c; d e f; g h i].
  template <typename T>
  arma::Mat<T> mat_3x3(T a, T b, T c, T d, T e, T f, T g, T h, T i) {
    arma::Mat<T> m(3,3);
    m(0,0) = a; m(0,1) = b; m(0,2) = c;
    m(1,0) = d; m(1,1) = e; m(1,2) = f;
    m(2,0) = g; m(2,1) = h; m(2,2) = i;
    return m;
  }

  //! Constructs a length-1 vector [a].
  template <typename T>
  arma::Col<T> vec_1(T a) {
    arma::Col<T> v(1);
    v[0] = a;
    return v;
  }

  //! Constructs a length-2 vector [a b].
  template <typename T>
  arma::Col<T> vec_2(T a, T b) {
    arma::Col<T> v(2);
    v[0] = a;
    v[1] = b;
    return v;
  }

  //! Constructs a length-3 vector [a b c].
  template <typename T>
  arma::Col<T> vec_3(T a, T b, T c) {
    arma::Col<T> v(3);
    v[0] = a;
    v[1] = b;
    v[2] = c;
    return v;
  }

  //! Constructs a sequence [from, from+1, ..., to-1].
  uvec sequence(size_t from, size_t to);

  // Serialization
  //============================================================================

  namespace impl {

    //! Read in a vector of values [val1,val2,...], ignoring an initial space
    //! if necessary.
    template <typename VecType, typename CharT>
    void read_vec_(std::basic_istream<CharT>& in, VecType& v) {
      CharT c;
      typedef typename VecType::value_type value_type;
      value_type val;
      std::vector<value_type> tmpv;
      in.get(c);
      if (c == ' ')
        in.get(c);
      assert(c == '[');
      if (in.peek() != ']') {
        do {
          if (!(in >> val))
            assert(false);
          tmpv.push_back(val);
          if (in.peek() == ',')
            in.ignore(1);
        } while (in.peek() != ']');
      }
      in.ignore(1);
      v = arma::conv_to<arma::Col<value_type> >::from(tmpv);
    }

    template <typename ArmaType>
    oarchive& operator_ll_(oarchive& a, const ArmaType& m) {
      a << m.n_rows << m.n_cols;
      typedef typename ArmaType::value_type value_type;
      const value_type* it  = m.begin();
      const value_type* end = m.end();
      for(; it != end; ++it)
        a << *it;
      return a;
    }

    template <typename VecType>
    iarchive& operator_gg_(iarchive& a, VecType& m) {
      size_t n_rows, n_cols;
      a >> n_rows >> n_cols;
      m.set_size(n_rows, n_cols);
      typedef typename VecType::value_type value_type;
      value_type* it  = m.begin();
      value_type* end = m.end();
      for(; it != end; ++it)
        a >> *it;
      return a;
    }

  } // namespace impl

  //! Read in a vector of values [val1,val2,...], ignoring an initial space
  //! if necessary.
  //! \todo Can we overload operator>> for this?  I tried but didn't get it to
  //!       work.
  template <typename T, typename CharT>
  void read_vec(std::basic_istream<CharT>& in, arma::Col<T>& v) {
    impl::read_vec_(in, v);
  }

  //! Read in a vector of values [val1,val2,...], ignoring an initial space
  //! if necessary.
  template <typename T, typename CharT>
  void read_vec(std::basic_istream<CharT>& in, arma::Row<T>& v) {
    impl::read_vec_(in, v);
  }

  // Joseph B.: I added this since the functions for arma::Mat were not
  //            identified by the compiler for arma::Col types.
  template <typename T>
  oarchive& operator<<(oarchive& a, const arma::Col<T>& m) {
    return impl::operator_ll_(a,m);
  }

  template <typename T>
  iarchive& operator>>(iarchive& a, arma::Col<T>& m) {
    return impl::operator_gg_(a,m);
  }

  template <typename T>
  oarchive& operator<<(oarchive& a, const arma::Row<T>& m) {
    return impl::operator_ll_(a,m);
  }

  template <typename T>
  iarchive& operator>>(iarchive& a, arma::Row<T>& m) {
    return impl::operator_gg_(a,m);
  }

  template <typename T>
  oarchive& operator<<(oarchive& a, const arma::Mat<T>& m) {
    return impl::operator_ll_(a,m);
  }

  template <typename T>
  iarchive& operator>>(iarchive& a, arma::Mat<T>& m) {
    return impl::operator_gg_(a,m);
  }

  // Functions to match sparse linear algebra interface
  //============================================================================

  //! Dense vector += dense matrix * dense vector
  /*
  template <typename T>
  void
  gemv(const arma::Mat<T>& m, const arma::Col<T>& v, arma::Col<T>& out) {
    out += m * v;
  }
  */

} // namespace sill

namespace arma {

  //! Read vector from string.
  template <typename T, typename CharT, typename Traits>
  std::basic_istream<CharT, Traits>&
  operator>>(std::basic_istream<CharT, Traits>& in, Col<T>& v) {
    sill::read_vec(in, v);
    return in;
  }

  //! Read matrix from string.
  template <typename T, typename CharT, typename Traits>
  std::basic_istream<CharT, Traits>&
  operator>>(std::basic_istream<CharT, Traits>& in, Mat<T>& v) {
    assert(false); // TO BE IMPLEMENTED
    return in;
  }

}

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_ARMADILLO_HPP
