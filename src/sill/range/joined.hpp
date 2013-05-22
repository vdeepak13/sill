#ifndef SILL_JOINED_HPP
#define SILL_JOINED_HPP

#include <boost/range/iterator_range.hpp>

#include <sill/iterator/join_iterator.hpp>

namespace sill {

  //! \ingroup range_adapters
  template <typename Range1, typename Range2>
  boost::iterator_range<
    join_iterator< typename boost::range_iterator<const Range1>::type,
                   typename boost::range_iterator<const Range2>::type > >
  make_joined(const Range1& r1, const Range2& r2) {
    return boost::make_iterator_range
      (make_join_iterator(boost::begin(r1), boost::end(r1), boost::begin(r2)),
       make_join_iterator(boost::end(r1),   boost::end(r1), boost::end(r2)));
  }
  
}

#endif
