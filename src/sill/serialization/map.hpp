#ifndef SILL_SERIALIZE_MAP_HPP
#define SILL_SERIALIZE_MAP_HPP

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/pair.hpp>

#include <iterator>
#include <map>

namespace sill {

  //! Serializes a map. \relates oarchive
  template <typename T, typename U>
  oarchive& operator<<(oarchive& ar, const std::map<T, U>& map) {
    ar.serialize_range(map.begin(), map.end(), map.size());
    return ar;
  }

  //! Deserializes a map. \relates iarchive
  template <typename T, typename U>
  iarchive& operator>>(iarchive& ar, std::map<T, U>& map) {
    map.clear();
    ar.deserialize_range<std::pair<T, U>>(std::inserter(map, map.end()));
    return ar;
  }

} // namespace sill

#endif
