#ifndef SILL_MATH_UBLAS_DETERMINANT_HPP
#define SILL_MATH_UBLAS_DETERMINANT_HPP

#include <vector>
#include <algorithm>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/lapack/gesv.hpp>

namespace boost { namespace numeric { namespace ublas
{

  //! Computes the determinant of a dense matrix
  //! \ingroup math_ublas
  template <typename E>
  typename E::value_type det(const matrix_expression<E>& ae) {
    typedef typename E::value_type value_type;
    matrix<value_type, column_major> a(ae());
    std::vector<int> ipiv(std::min(a.size1(), a.size2()));
    int result = boost::numeric::bindings::lapack::getrf(a, ipiv);
    assert(result>=0);

    // compute the sign of the permutation matrix returned by getrf
    // see http://nacad.ufrj.br/sgi/007-4325-001/sgi_html/ch03.html
    std::size_t count = 0;
    for(std::size_t i=0; i<ipiv.size(); i++) 
      count += (ipiv[i] != int(i+1));

    // compute the product manually for now
    typename E::value_type d = 1;
    for(std::size_t i = 0; i < a.size1(); i++)
      d *= a(i,i);
    return d * ((count%2) ? -1 : 1);
  }

  //! Computes the log-determinant of a dense matrix
  //! \ingroup math_ublas
  template <typename E>
  typename E::value_type logdet(const matrix_expression<E>& ae) {
    using std::log;
    using std::abs;
    typedef typename E::value_type value_type;
    matrix<value_type, column_major> a(ae());
    std::vector<int> ipiv(std::min(a.size1(), a.size2()));
    int result = boost::numeric::bindings::lapack::getrf(a, ipiv);
    assert(result>=0);

    // compute the sign of the permutation matrix returned by getrf
    // see http://nacad.ufrj.br/sgi/007-4325-001/sgi_html/ch03.html
    std::size_t count = 0;
    for(std::size_t i=0; i<ipiv.size(); i++) 
      count += (ipiv[i] != int(i+1));

    typename E::value_type ld = 0;
    for(std::size_t i = 0; i<a.size1(); i++) {
      if (a(i,i)<0) count++;
      ld += log(abs(a(i,i)));
    }

    ld += ((count%2) ? log(value_type(-1)) : 0); // log(-1) is sort of stupid
    return ld;
  }

} } } // namespaces

#endif
