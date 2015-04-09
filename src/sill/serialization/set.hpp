#ifndef SILL_SERIALIZE_SET_HPP
#define SILL_SERIALIZE_SET_HPP

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

#include <iterator>
#include <set>

namespace sill {

  //! Serializes a set. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const std::set<T>& set){
    ar.serialize_range(set.begin(), set.end(), set.size());
    return ar;
  }

  //! Deserializes a set. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, std::set<T>& set) {
    set.clear();
    ar.deserialize_range<T>(std::inserter(set, set.end()));
    return ar;
  }

} // namespace sill

#endif
