#ifndef PRL_MATH_UBLAS_CHOL_HPP
#define PRL_MATH_UBLAS_CHOL_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/lapack/posv.hpp>

namespace boost { namespace numeric { namespace ublas
{
  /**
   * Computes the Cholesky factorization of a symmetric positive matrix.
   * @param upper if true, returns the upper Cholesky factorization U:
   *              A = U^T * U.
   *              if false,returns the lower Cholesky factorization L:
   *              A = L * L^T.
   * @throw std::domain_error if the factorization fails.
   *
   * \ingroup math_ublas
   */
  template <typename E>
  matrix<typename E::value_type, column_major>
  chol(const matrix_expression<E>& e, bool upper = true) {
    matrix<typename E::value_type, column_major> a(e());
    int result = boost::numeric::bindings::lapack::potrf(upper ? 'U': 'L', a);
    assert(result>=0);
    if (result != 0) {
      throw std::domain_error("Cholesky factorization failed.");
      /*
      using namespace std;
      cerr << "Cholesky factorization failed; " 
           << "matrix is not positive definite:" << endl
           << e() << endl;
      assert(false);
      */
    }
    // zero out the entries that were not used during the decomposition
    // this may not not be the most efficient implementation
    if(upper)
      for(size_t i = 1; i < a.size1(); i++)
        for(size_t j = 0; j < i; j++)
          a(i,j) = 0;
    else
      for(size_t i = 0; i < a.size1(); i++)
        for(size_t j = i+1; j < a.size2(); j++)
          a(i,j) = 0;
    return a;
  }
  
} } } // namespaces


#endif
