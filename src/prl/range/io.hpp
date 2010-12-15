#ifndef PRL_RANGE_IO_HPP
#define PRL_RANGE_IO_HPP

#include <iosfwd>
#include <boost/range/iterator_range.hpp>

#include <prl/stl_io.hpp>

namespace prl {

  template <typename It>
  std::ostream& operator<<(std::ostream& out, 
                           const boost::iterator_range<It>& range) {
    print_range(out, range, '(', ' ', ')');
    return out;
  }

}

#endif
