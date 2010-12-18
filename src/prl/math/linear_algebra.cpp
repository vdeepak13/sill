#include <sill/math/linear_algebra.hpp>

namespace sill {

  //! Computes the log-determinant of a matrix
  double logdet(const itpp::mat& a) {
    mat l = chol(a);
    return 2*sum(log(diag(l)));
  }

} // namespace sill
