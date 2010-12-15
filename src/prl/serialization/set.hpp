#ifndef PRL_SERIALIZE_SET_HPP
#define PRL_SERIALIZE_SET_HPP

#include <set>

#include <prl/serialization/iarchive.hpp>
#include <prl/serialization/oarchive.hpp>
#include <prl/serialization/iterator.hpp>

namespace prl {

  /**
    Serializes a set
    Returns true on success, false on failure  */
  template <typename T>
  oarchive& operator<<(oarchive& a, const std::set<T>& vec){
    serialize_iterator(a,vec.begin(),vec.end(), vec.size());
    return a;
  }

  /**
    deserializes a set
    Returns true on success, false on failure  */
  template <typename T>
  iarchive& operator>>(iarchive& a, std::set<T>& vec){
    vec.clear();
    deserialize_iterator<T>(a, std::inserter(vec,vec.end()));
    return a;
  }

} // namespace prl

#endif //PRL_SERIALIZE_SET_HPP
