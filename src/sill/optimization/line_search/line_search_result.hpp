#ifndef SILL_LINE_SEARCH_RESULT_HPP
#define SILL_LINE_SEARCH_RESULT_HPP

#include <iostream>

namespace sill {

  /**
   * A struct that represents a step along a line and the associated
   * objective value.
   */
  template <typename RealType>
  struct line_search_result {
    RealType step;
    RealType value;
    explicit line_search_result(RealType step, RealType value)
      : step(step), value(value) { }
    void reset(RealType step, RealType value) {
      this->step = step;
      this->value = value;
    }
  }; // struct line_search_result

  //! \relates line_search_result
  template <typename RealType>
  std::ostream& 
  operator<<(std::ostream& out, const line_search_result<RealType>& r) {
    out << r.step << ':' << r.value;
    return out;
  }
  
} // namespace sill

#endif
