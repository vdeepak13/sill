#ifndef PRL_MATH_UBLAS_SOLVE_HPP
#define PRL_MATH_UBLAS_SOLVE_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/lapack/posv.hpp>

namespace boost { namespace numeric { namespace ublas
{

  //! Solves a system of equations AX=B where A is positive definite
  //! \ingroup math_ublas
  template <typename E1, typename E2>
  matrix<typename E1::value_type, column_major>
  spd_solve(const matrix_expression<E1>& ea,
            const matrix_expression<E2>& eb) {
    typedef typename E1::value_type value_type;
    matrix<value_type, column_major> a = ea();
    matrix<value_type, column_major> b = eb();
    if (!isempty(b)) {
      int result = boost::numeric::bindings::lapack::posv('U', a, b);
      if (result!=0) {
        using namespace std;
        cerr << "Cholesky factorization failed; " 
             << "matrix is not positive definite:" << endl
             << ea() << endl;
        /* assert(false); */
      }
    }
    return b;
  }

  //! Solves a system of equations AX=b where A is positive definite
  //! \ingroup math_ublas
  template <typename ME, typename VE>
  vector<typename ME::value_type>
  spd_solve(const matrix_expression<ME>& ea,
            const vector_expression<VE>& eb) {
    typedef typename ME::value_type value_type;
    matrix<value_type, column_major> a = ea();
    matrix<value_type, column_major> b(eb().size(),1);
    b.column(0) = eb();
    if (!isempty(b)) {
      int result = boost::numeric::bindings::lapack::posv('U', a, b);
      if (result!=0) {
        using namespace std;
        cerr << "Cholesky factorization failed; " 
             << "matrix is not positive definite:" << endl
             << ea() << endl;
        /* assert(false); */
      }
    }
    return b.column(0);
  }


} } } // namespaces

#endif
