#ifndef SILL_RANGE_TRANSFORMED_HPP
#define SILL_RANGE_TRANSFORMED_HPP

#include <boost/range/iterator_range.hpp>
#include <boost/iterator/transform_iterator.hpp>

namespace sill {

  //! \ingroup range_adapters
  template <typename Range, typename F>
  boost::iterator_range<
    boost::transform_iterator<
      F, typename boost::range_iterator<const Range>::type> >
  make_transformed(const Range& range, F f) {
    return boost::make_iterator_range
      (boost::make_transform_iterator(boost::begin(range), f),
       boost::make_transform_iterator(boost::end(range), f));
  }
  
}

#endif
