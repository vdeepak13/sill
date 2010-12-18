#ifndef SILL_CONVERTED_HPP
#define SILL_CONVERTED_HPP

#include <boost/range/iterator_range.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <sill/functional.hpp>

namespace sill {

  //! \ingroup range_adapters
  template <typename T, typename Range>
  boost::iterator_range<
    boost::transform_iterator<
      converter<T>,
      typename boost::range_iterator<const Range>::type> >
  make_converted(const Range& range) {
    return boost::make_iterator_range
      (boost::make_transform_iterator(boost::begin(range), converter<T>()),
       boost::make_transform_iterator(boost::end(range), converter<T>()));
  }

}

#endif
