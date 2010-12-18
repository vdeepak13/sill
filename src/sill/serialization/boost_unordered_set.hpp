#ifndef SILL_SERIALIZE_BOOST_UNORDERED_SET_HPP
#define SILL_SERIALIZE_BOOST_UNORDERED_SET_HPP

#include <boost/unordered_set.hpp>

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/iterator.hpp>

namespace sill {

  /**
    Serializes a set
    Returns true on success, false on failure  */
  template <typename T>
  oarchive& operator<<(oarchive& a, const boost::unordered_set<T>& set){
    serialize_iterator(a, set.begin(), set.end(), set.size());
    return a;
  }

  /**
    deserializes a set
    Returns true on success, false on failure  */
  template <typename T>
  iarchive& operator>>(iarchive& a, boost::unordered_set<T>& set){
    set.clear();
    deserialize_iterator<T>(a, std::inserter(set, set.end()));
    return a;
  }

} // namespace sill

#endif //PRL_SERIALIZE_SET_HPP
