#ifndef SILL_MATH_UBLAS_EIGENVALUES_HPP
#define SILL_MATH_UBLAS_EIGENVALUES_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>

namespace boost { namespace numeric { namespace ublas 
{

  /**
   * Computes the eigenvalues and eigenvectors of a symmetric matrix
   * q * d * trans(q)
   * Returns true if the operation succeeded
   * \ingroup math_ublas
   */
  template <typename E>
  bool eig(const matrix_expression<E>& ae,
           vector<typename E::value_type>& d,
           matrix<typename E::value_type, column_major>& q) {
    typedef typename E::value_type value_type;
    assert(is_symmetric(ae()));
    d.resize(ae().size1());
    q.resize(ae().size1(), ae().size2());
    q = ae();
    int result = 
      boost::numeric::bindings::lapack::syev
      ('V', 'U', q, d, boost::numeric::bindings::lapack::optimal_workspace());
    assert(result>=0);
    if (result) {
      using namespace std;
      cerr << "syev failed to converge: " << ae() << endl;
      return false;
    }
    return true;
  }

  // todo: function that only computes the eigenvalues

} } } // namespaces

#endif
