#ifndef SILL_INDEX_MAP_HPP
#define SILL_INDEX_MAP_HPP

#include <boost/range/value_type.hpp>

#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Returns an object that maps the elements of a range (v_0, ..., v_n-1) to 
  //! values 0, ..., n-1.
  //! \relates map
  template <typename R>
  std::map<typename R::value_type, std::size_t>
  make_index_map(const R& values) {
    concept_assert((InputRange<R>));
    typedef typename boost::range_value<R>::type value_type;
    std::size_t index = 0;
    std::map<value_type, std::size_t> index_map;
    foreach(value_type v, values)
      index_map[v] = index++;
    return index_map;
  }


}

#include <sill/macros_undef.hpp>

#endif
