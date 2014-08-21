#ifndef SILL_HYBRID_GRADIENT_METHOD_HPP
#define SILL_HYBRID_GRADIENT_METHOD_HPP

#include <sill/optimization/gradient_method.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for creating a hybrid of multiple optimization algorithms.
   *
   * This class currently only supports running one algorithm for a fixed
   * number of iterations and then running a second algorithm.
   * E.g., you might wish to run stochastic_gradient to get close to an
   * optimum quickly before running gradient_descent to get to the optimum.
   */
  template </*RIGHT HERE NOW*/>
  class hybrid_gradient_method
    : public gradient_method</*RIGHT HERE NOW*/> {
  }; // class hybrid_gradient_method

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef _SILL_HYBRID_GRADIENT_METHOD_HPP_
