#ifndef SILL_ARMADILLO_HPP
#define SILL_ARMAILLOD_HPP

#include <armadillo>

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
  
} // namespace sill

#endif
