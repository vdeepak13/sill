#ifndef SILL_MATH_OPERATIONS_HPP
#define SILL_MATH_OPERATIONS_HPP

#include <cmath>

namespace sill {

  //! \addtogroup math_operations
  //! @{

  //! Square a value.
  template <typename T> 
  T sqr(const T& value) { return value*value; }

  //! Round a value to the nearest integer.
  //! Note *.5 is rounded to *, but -*.5 is rounded to -(*+1).
  template <typename T>
  T round(T value) {
    return ceil(value - .5);
  }

  //! @}

} // namespace sill

#endif // SILL_MATH_OPERATIONS_HPP
