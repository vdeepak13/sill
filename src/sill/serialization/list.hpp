#ifndef SILL_SERIALIZE_LIST_HPP
#define SILL_SERIALIZE_LIST_HPP

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

#include <iterator>
#include <list>

namespace sill {

  //! Serializes a list. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& ar, const std::list<T>& list) {
    ar.serialize_range(list.begin(), list.end(), list.size());
    return ar;
  }

  //! Deserializes a list. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& ar, std::list<T>& list) {
    list.clear();
    ar.deserialize_range<T>(std::back_inserter(list));
    return ar;
  }

} // namespace sill

#endif

