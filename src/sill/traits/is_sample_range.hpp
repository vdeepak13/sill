#ifndef SILL_IS_SAMPLE_RANGE_HPP
#define SILL_IS_SAMPLE_RANGE_HPP

#include <type_traits>
#include <utility>

namespace sill {

  /**
   * Defined to be the true type if the given object represents a range
   * over weighted samples.
   * \tparam Range object tested
   * \tparam Sample sample type
   * \tparam Weight weight type
   */
  template <typename Range, typename Sample, typename Weight>
  struct is_sample_range
    : public std::is_convertible<typename Range::value_type,
                                 std::pair<Sample, Weight>> { };

} // namespace sill

#endif
