#ifndef PRL_SERIALIZE_BOOST_UNORDERED_SET_HPP
#define PRL_SERIALIZE_BOOST_UNORDERED_SET_HPP

#include <boost/unordered_set.hpp>

#include <prl/serialization/iarchive.hpp>
#include <prl/serialization/oarchive.hpp>
#include <prl/serialization/iterator.hpp>

namespace prl {

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

} // namespace prl

#endif //PRL_SERIALIZE_SET_HPP
