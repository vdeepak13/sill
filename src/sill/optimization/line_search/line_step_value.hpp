#ifndef SILL_LINE_STEP_VALUE_HPP
#define SILL_LINE_STEP_VALUE_HPP

#include <iostream>

namespace sill {

  /**
   * A struct that represents a step along a line and the associated
   * objective value.
   */
  template <typename RealType>
  struct line_step_value {
    RealType step;
    RealType value;
    explicit line_step_value(RealType step, RealType value)
      : step(step), value(value) { }
  }; // struct line_step_value

  //! \relates line_step_value
  template <typename RealType>
  std::ostream& 
  operator<<(std::ostream& out, const line_step_value<RealType>& p) {
    out << p.step << ':' << p.value;
    return out;
  }
  
} // namespace sill

#endif
