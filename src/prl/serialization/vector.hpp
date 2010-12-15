#ifndef PRL_SERIALIZE_VECTOR_HPP
#define PRL_SERIALIZE_VECTOR_HPP
#include <prl/serialization/iarchive.hpp>
#include <prl/serialization/oarchive.hpp>
#include <prl/serialization/iterator.hpp>
#include <vector>
namespace prl {
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
} // namespace prl

#endif //PRL_SERIALIZE_VECTOR_HPP
