#ifndef SILL_ARMADILLO_HPP
#define SILL_ARMADILLO_HPP

#include <armadillo>

#include <sill/base/stl_util.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/serialization/iterator.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  template <typename Ref> class forward_range;

  // bring Armadillo's types and functions into the sill namespace

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
  
  // serialization
  // Joseph B.: I added this since the functions for arma::Mat were not
  //            identified by the compiler for arma::Col types.
  template <typename T>
  oarchive& operator<<(oarchive& a, const arma::Col<T>& m) {
    a << m.n_elem;
    const T* it  = m.begin();
    const T* end = m.end();
    for(; it != end; ++it)
      a << *it;
    return a;
  }

  template <typename T>
  iarchive& operator>>(iarchive& a, arma::Col<T>& m) {
    size_t n_elem;
    a >> n_elem;
    m.set_size(n_elem);
    T* it  = m.begin();
    T* end = m.end();
    for(; it != end; ++it)
      a >> *it;
    return a;
  }

  // serialization
  template <typename T>
  oarchive& operator<<(oarchive& a, const arma::Mat<T>& m) {
    a << m.n_rows << m.n_cols;
    const T* it  = m.begin();
    const T* end = m.end();
    for(; it != end; ++it)
      a << *it;
    return a;
  }

  template <typename T>
  iarchive& operator>>(iarchive& a, arma::Mat<T>& m) {
    size_t n_rows, n_cols;
    a >> n_rows >> n_cols;
    m.set_size(n_rows, n_cols);
    T* it  = m.begin();
    T* end = m.end();
    for(; it != end; ++it)
      a >> *it;
    return a;
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
        result(span(n, n + v.size() - 1)) = v;
      }
      n += v.size();
    }
    return result;
  }

  //! Read in a vector of values [val1,val2,...], ignoring an initial space
  //! if necessary.
  //! \todo Can we overload operator>> for this?  I tried but didn't get it to
  //!       work.
  template <typename T, typename CharT>
  void read_vec(std::basic_istream<CharT>& in, arma::Col<T>& v) {
    CharT c;
    T val;
    std::vector<T> tmpv;
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
    v = arma::conv_to<arma::Col<T> >::from(tmpv);
  }

  /**
   * Compare two vectors.
   * @return -1 if a is smaller or +1 if b is smaller.
   *         If a,b have the same size, return lexigraphic_compare(a,b).
   */
  template <typename T>
  int compare(const arma::Col<T>& a, const arma::Col<T>& b) {
    if (a.size() < b.size())
      return -1;
    if (a.size() > b.size())
      return 1;
    return lexigraphic_compare(a,b);
  }

  /**
   * Compare two matrices.
   * @return -1 if a is smaller in n_rows, then n_cols; +1 if b is smaller.
   *         If a,b have the same size, return lexigraphic_compare(a,b).
   */
  template <typename T>
  int compare(const arma::Mat<T>& a, const arma::Mat<T>& b) {
    if (a.n_rows < b.n_rows)
      return -1;
    if (a.n_rows > b.n_rows)
      return 1;
    if (a.n_cols < b.n_cols)
      return -1;
    if (a.n_cols > b.n_cols)
      return 1;
    return lexigraphic_compare(forward_range<T>(a.begin(),a.end()),
                               forward_range<T>(b.begin(),b.end()));
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
