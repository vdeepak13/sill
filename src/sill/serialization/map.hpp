#ifndef SILL_SERIALIZE_MAP_HPP
#define SILL_SERIALIZE_MAP_HPP

#include <map>

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/iterator.hpp>

namespace sill {

  /** Serializes a map
      Returns true on success, false on failure  */
  template <typename T, typename U>
  oarchive& operator<<(oarchive& a, const std::map<T,U>& vec){
    serialize_iterator(a,vec.begin(),vec.end(), vec.size());
    return a;
  }

  /** deserializes a map
      Returns true on success, false on failure  */
  template <typename T, typename U>
  iarchive& operator>>(iarchive& a, std::map<T,U>& vec){
    vec.clear();
    deserialize_iterator<std::pair<T,U> >(a, std::inserter(vec,vec.end()));
    return a;
  }
} // namespace sill

#endif //PRL_SERIALIZE_MAP_HPP
