#ifndef SILL_MATH_OPERATIONS_HPP
#define SILL_MATH_OPERATIONS_HPP

#include <cmath>

#include <sill/global.hpp>

namespace sill {

  //! \addtogroup math_operations
  //! @{

  //! Return true iff the values are equal.
  //! This is analogous to the equal() method for Armadillo vectors.
  bool equal(size_t a, size_t b);

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
