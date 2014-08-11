#ifndef SILL_SERIALIZE_SET_HPP
#define SILL_SERIALIZE_SET_HPP

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/range.hpp>

#include <set>

namespace sill {

  //! Serializes a set. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& a, const std::set<T>& set){
    serialize_range(a,set.begin(), set.end(), set.size());
    return a;
  }

  //! Deserializes a set. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& a, std::set<T>& set) {
    set.clear();
    deserialize_range<T>(a, std::inserter(set, set.end()));
    return a;
  }

} // namespace sill

#endif
