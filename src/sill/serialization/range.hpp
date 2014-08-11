#ifndef SILL_SERIALIZE_RANGE_HPP
#define SILL_SERIALIZE_RANGE_HPP

#include <iterator>

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

namespace sill {

  /**
   * Serializes the contents between the iterators begin and end.
   * This version prefers the availability of RandomAccessIterator since it needs
   * a distance between the begin and end iterator.
   * This function as implemented will work for other input iterators
   * but is extremely inefficient.
   */
  template <typename RandomAccessIterator>
  void serialize_range(oarchive& a,
                       RandomAccessIterator begin,
                       RandomAccessIterator end) {
    size_t vsize = std::distance(begin, end);
    a << vsize;
    for(; begin != end; ++begin) {
      a << *begin;
    }
  }

  /**
   * Serializes the contents between the iterators begin and end.
   * This version takes all InputIterator types, but takes a "count" for
   * efficiency. This count is checked and will return failure if the number
   * of elements serialized does not match the count.
   */
  template <typename InputIterator>
  void serialize_range(oarchive& a,
                       InputIterator begin,
                       InputIterator end,
                       size_t vsize) {
    a << vsize;
    //store each element
    size_t count = 0;
    for(; begin != end; ++begin) {
      ++count;
      a << *begin;
    }
    // fail if count does not match
    assert(count == vsize);
  }

  /**
   * The accompanying function to serialize_range()
   * Reads elements from the stream and send it to the output iterator.
   * \tparam T The type of object to deserialize. This is necessary for
   *           instance for the map type. The map<T,U>::value_type
   *           is pair<const T,U> which is not useful since we cannot
   *           assign to it. In this case, T=pair<T,U>.
   */
  template <typename T, typename OutputIterator>
  void deserialize_range(iarchive& a, OutputIterator result) {
    // get the number of elements to deserialize
    size_t length = 0;
    a >> length;
    
    // iterate through and send to the output iterator
    for (size_t i = 0; i < length; ++i){
      T v;
      a >> v;
      *result = v;
      ++result;
    }
  }

} // namespace sill

#endif

