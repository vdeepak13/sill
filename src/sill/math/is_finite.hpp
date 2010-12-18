#ifndef SILL_IS_FINITE_HPP
#define SILL_IS_FINITE_HPP

#include <limits>
#include <cmath>

namespace sill {

  //! \addtogroup math_number

  //! Returns true if the given number is positive and finite
  //! also see boost/numeric/conversion/bounds.hpp
  template <typename T>
  bool is_positive_finite(T x) {
    if (std::numeric_limits<T>::has_infinity)
      return x>T(0) && x!= std::numeric_limits<T>::infinity();
    else
      return x>T(0);
  }

  //! Returns true if the given number is finite
  template <typename T>
  bool is_finite(T x) {
    if (std::numeric_limits<T>::has_infinity) {
      using std::abs;
      return abs(x) != std::numeric_limits<T>::infinity();
    } else
      return true;
  }

} // namespace sill

#endif

