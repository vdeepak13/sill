#ifndef PRL_RANGE_REVERSED_HPP
#define PRL_RANGE_REVERSED_HPP

#include <boost/range/iterator_range.hpp>
#include <boost/iterator/reverse_iterator.hpp>

namespace prl {

  //! \ingroup range_adapters
  template <typename Range>
  boost::iterator_range<
    boost::reverse_iterator<typename boost::range_iterator<const Range>::type> >
  make_reversed(const Range& range) {
    return boost::make_iterator_range
      (boost::make_reverse_iterator(boost::end(range)),
       boost::make_reverse_iterator(boost::begin(range)));
  }
  
}

#endif
