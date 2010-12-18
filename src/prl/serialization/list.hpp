#ifndef SILL_SERIALIZE_LIST_HPP
#define SILL_SERIALIZE_LIST_HPP
#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/iterator.hpp>
#include <list>
namespace sill {

  /** Serializes a list
      Returns true on success, false on failure  */
  template <typename T>
  oarchive& operator<<(oarchive& a, const std::list<T>& vec){
    serialize_iterator(a,vec.begin(), vec.end());
    return a;
  }

  /** deserializes a list
      Returns true on success, false on failure  */
  template <typename T>
  iarchive& operator>>(iarchive& a, std::list<T>& vec){
    vec.clear();
    deserialize_iterator<T>(a, std::inserter(vec, vec.end()));
    return a;
  }
} // namespace sill
#endif //PRL_SERIALIZE_LIST_HPP
