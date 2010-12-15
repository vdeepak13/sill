#ifndef PRL_LINEAR_ALGEBRA_HPP
#define PRL_LINEAR_ALGEBRA_HPP

#include <itpp/base/algebra/cholesky.h>
#include <itpp/itbase.h>

#include <prl/math/vector.hpp>
#include <prl/math/matrix.hpp>

namespace prl {
  

  // Special matrices
  //============================================================================

  // Bring in declarations for functions that cannot be found by 
  // argument-dependent lookup
  using itpp::ones;
  using itpp::ones_b;
  using itpp::ones_i;
  using itpp::ones_c;

  using itpp::zeros;
  using itpp::zeros_b;
  using itpp::zeros_i;
  using itpp::zeros_c;

  //! \ingroup math_linalg
  inline itpp::mat identity(size_t size) { return itpp::eye(size); }
  //! \ingroup math_linalg
  inline itpp::bmat identity_b(size_t size) { return itpp::eye_b(size); }
  //! \ingroup math_linalg
  inline itpp::imat identity_i(size_t size) { return itpp::eye_i(size); }
  //! \ingroup math_linalg
  inline itpp::cmat identity_c(size_t size) { return itpp::eye_c(size); }

  template <typename T> 
  void identity(size_t size, itpp::Mat<T>& m) { itpp::eye(size, m); }

  using itpp::conference;
  using itpp::hadamard;
  using itpp::impulse;
  using itpp::jacobsthal;
  using itpp::linspace;
  using itpp::outer_product;
  using itpp::toeplitz;
  using itpp::zigzag_space;

  using itpp::vec_1;
  using itpp::vec_2;
  using itpp::vec_3;

  using itpp::mat_1x1;
  using itpp::mat_1x2;
  using itpp::mat_2x1;
  using itpp::mat_2x2;
  using itpp::mat_1x3;
  using itpp::mat_3x1;
  using itpp::mat_2x3;
  using itpp::mat_3x2;
  using itpp::mat_3x3;

  /*
  //! Returns a vectors filled with a given value
  template <typename T>
  vector<T> scalars(size_t n, T value);

  //! Returns a matrix filled 
  template <typename T>
  vector<T> scalars(size_t m, size_t n, T value);
  */

  // log-determinants
  //============================================================================
  //! Computes the log-determinant of a positive definite matrix
  //! \ingroup math_linalg
  double logdet(const itpp::mat& a);

} // namespace prl

#endif
