#ifndef SILL_SERIALIZE_VECTOR_HPP
#define SILL_SERIALIZE_VECTOR_HPP

#include <vector>

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/iterator.hpp>

namespace sill {

  /**
    Serializes a vector
    Returns true on success, false on failure  */
  template <typename T>
  oarchive& operator<<(oarchive& a, const std::vector<T>& vec){
    serialize_iterator(a,vec.begin(), vec.end());
    return a;
  }

  /**
    deserializes a vector
    Returns true on success, false on failure  */
  template <typename T>
  iarchive& operator>>(iarchive& a, std::vector<T>& vec){
    vec.clear();
    deserialize_iterator<T>(a, std::inserter(vec, vec.end()));
    return a;
  }

} // namespace sill

#endif //PRL_SERIALIZE_VECTOR_HPP
