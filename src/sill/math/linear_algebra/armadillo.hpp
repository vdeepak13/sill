#ifndef SILL_ARMADILLO_HPP
#define SILL_ARMADILLO_HPP

#include <armadillo>

#include <sill/range/forward_range.hpp>
#include <sill/serialization/iterator.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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
    foreach(const arma::Col<T>& v, vectors) n += v.n_elem;
    arma::Col<T> result(n);

    // assign the vectors to the right indices
    n = 0;
    foreach(const arma::Col<T>& v, vectors) {
      if (!v.is_empty()) {
        result(span(n, n+v.n_elem-1)) = v;
      }
      n += v.n_elem;
    }
    return result;
  }

  //! Read in a vector of values [val1,val2,...], ignoring an initial space
  //! if necessary.
  //! \todo Can we overload operator<< for this?  I tried but didn't get it to
  //!       work.
  template <typename T>
  static void read_vec(std::istream& in, arma::Col<T>& v) {
    char c;
    T val;
    v.resize(0);
    in.get(c);
    if (c == ' ')
      in.get(c);
    assert(c == '[');
    if (in.peek() != ']') {
      do {
        if (!(in >> val))
          assert(false);
        v.insert(v.size(),val);
        if (in.peek() == ',')
          in.ignore(1);
      } while (in.peek() != ']');
    }
    in.ignore(1);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_ARMADILLO_HPP
