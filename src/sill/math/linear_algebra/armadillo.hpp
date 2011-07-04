#ifndef SILL_ARMADILLO_HPP
#define SILL_ARMAILLOD_HPP

#include <armadillo>

#include <sill/serialization/iterator.hpp>

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
    vec result(n);

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

} // namespace sill

#endif
