#ifndef SILL_MATH_UBLAS_IO_HPP
#define SILL_MATH_UBLAS_IO_HPP

#include <iosfwd>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace boost { namespace numeric { namespace ublas
{

  // Technically, we may be violating the one-definition rule here
  // if other programs include the standard <boost/numeric/ublas.io.hpp>
  // Is there a way around this issue, e.g., using anonymous namespaces
  //! \ingroup math_ublas
  template <typename Char, typename Traits, typename E>
  std::basic_ostream<Char, Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const vector_expression<E>& e) {
    std::size_t n = e().size();
    out << '[';
    for(std::size_t i = 0; i < n; i++) {
      if (i>0) out << ' ';
      out << e()[i];
    }
    out << ']';
    return out;
  }

  //! \ingroup math_ublas
  template <typename Char, typename Traits, typename E>
  std::basic_ostream<Char, Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const matrix_expression<E>& e) {
    std::size_t m = e().size1(), n = e().size2();
    out << '[';
    for(std::size_t i = 0; i < m; i++) {
      if (i>0) out << "; ";
      for(std::size_t j = 0; j < n; j++) {
        if (j>0) out << ' ';
        out << e()(i,j);
      }
    }
    out << ']';
    return out;
  }

} } } // namespaces

#endif
