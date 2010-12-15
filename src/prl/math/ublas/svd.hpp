#ifndef PRL_MATH_UBLAS_SVD_HPP
#define PRL_MATH_UBLAS_SVD_HPP

#include <algorithm>

#include <boost/numeric/bindings/lapack/gesvd.hpp>

namespace boost { namespace numeric { namespace ublas
{

  //! Computes the singular value decomposition of a dense matrix
  //! \ingroup math_ublas
  template <typename E>
  bool svd(const matrix_expression<E>& ae,
           matrix<typename E::value_type, column_major>& u, 
           vector<typename E::value_type>&               s,
           matrix<typename E::value_type, column_major>& v) {
    typedef typename E::value_type value_type;
    std::size_t minmn = std::min(ae().size1(), ae().size2());
    matrix<value_type, column_major> a(ae());
    matrix<value_type, column_major> vt(minmn, ae().size2());
    u.resize(a.size1(), minmn);
    s.resize(minmn);
    v.resize(a.size2(), minmn);
    int result = boost::numeric::bindings::lapack::gesvd(a, s, u, vt);
    if (result!=0) {
      using namespace std;
      cerr << "Singular value decomposition failed " << ae() << endl;
      return false;
    }
    return true;
  }
  
} } } // namespaces

#endif
