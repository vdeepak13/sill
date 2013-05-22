#ifndef SILL_RANGE_IO_HPP
#define SILL_RANGE_IO_HPP

#include <iosfwd>
#include <boost/range/iterator_range.hpp>

#include <sill/stl_io.hpp>

namespace sill {

  template <typename It>
  std::ostream& operator<<(std::ostream& out, 
                           const boost::iterator_range<It>& range) {
    print_range(out, range, '(', ' ', ')');
    return out;
  }

}

#endif
