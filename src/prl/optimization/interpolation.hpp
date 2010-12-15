
#ifndef PRL_OPTIMIZATION_INTERPOLATION_HPP
#define PRL_OPTIMIZATION_INTERPOLATION_HPP

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <prl/macros_def.hpp>

namespace prl {

  //! \addtogroup optimization
  //! @{

  /**
   * Interpolate a function f(x) via a cubic polynomial using
   * f(0), f(a), f'(0), f'(a).
   * @param x   query point
   * @param a   This must be non-zero.
   * @param f0  f(0)
   * @param fa  f(a)
   * @param d0  f'(0)
   * @param da  f'(a)
   * @return Estimate of f(x)
   */
  double interpolate_cubic_poly(double x, double a, double f0, double fa,
                                double d0, double da) {
    assert(a != 0);
    return f0 + x * (d0 + x *
                     ((((3/a) * (fa - f0)) - (da + 2*d0))/a
                      + x * ((((-2/a) * (fa - f0)) + (da + d0))/(a*a))));
  }

  //! @} group optimization

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_OPTIMIZATION_INTERPOLATION_HPP
