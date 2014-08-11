#ifndef SILL_SERIALIZE_LIST_HPP
#define SILL_SERIALIZE_LIST_HPP

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/range.hpp>

#include <list>

namespace sill {

  //! Serializes a list. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& a, const std::list<T>& list) {
    serialize_range(a,list.begin(), list.end());
    return a;
  }

  //! Deserializes a list. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& a, std::list<T>& list) {
    list.clear();
    deserialize_range<T>(a, std::inserter(list, list.end()));
    return a;
  }

} // namespace sill

#endif

