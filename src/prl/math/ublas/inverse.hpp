#ifndef PRL_MATH_UBLAS_INVERSE_HPP
#define PRL_MATH_UBLAS_INVERSE_HPP

#include <boost/numeric/bindings/lapack/posv.hpp>

#include <boost/numeric/ublas/matrix.hpp>

namespace boost { namespace numeric { namespace ublas
{
  /**
   * Computes the inverse of a positive definite matrix
   * using Cholesky factorization
   * \throw std::domain error if the matrix is not positive definite.
   * \ingroup math_ublas
   */
  template <typename T>
  matrix<T, column_major> spd_inv(const matrix<T, column_major>& a) {
    assert(issquare(a));
    if (isempty(a)) return a;
    matrix<T, column_major> a1 = a;
    matrix<T, column_major> b1 = identity_matrix<T>(a.size1());
    int result = boost::numeric::bindings::lapack::posv('U', a1, b1);
    assert(result >= 0);
    if (result!=0) {
      throw std::domain_error("Cholesky factorization failed.");
      /*
      using namespace std;
      cerr << "Cholesky factorization failed; " 
           << "matrix is not positive definite:" << endl
           << a << endl;
      assert(false); 
      */
    }
    return b1;
  }

} } } // namespaces

#endif
