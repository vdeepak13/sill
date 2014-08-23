#ifndef SILL_SLICE_HPP
#define SILL_SLICE_HPP

#include <sill/global.hpp>

#include <iostream>

namespace sill {
  
  /**
   * A simple class that represents a single slice of dataset.
   * A slice is an half-open interval [begin, end) over rows.
   */
  struct slice {
    size_t begin, end;

    slice() : begin(0), end(0) { }
    slice(size_t begin, size_t end) : begin(begin), end(end) { }
    size_t size() const { return end - begin; }
    bool empty() const { return begin == end; }
  };

  //! \relates slice
  std::ostream& operator<<(std::ostream& out, const slice& s) {
    out << '[' << s.begin << ',' << s.end << ')';
    return out;
  }

} // namespace sill

#endif
