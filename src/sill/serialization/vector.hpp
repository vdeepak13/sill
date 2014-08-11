#ifndef SILL_SERIALIZE_VECTOR_HPP
#define SILL_SERIALIZE_VECTOR_HPP

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>
#include <sill/serialization/range.hpp>

#include <vector>

namespace sill {

  //! Serializes a vector. \relates oarchive
  template <typename T>
  oarchive& operator<<(oarchive& a, const std::vector<T>& vec){
    serialize_range(a, vec.begin(), vec.end());
    return a;
  }

  //! Serializes a vector<bool>. \relates oarchive
  inline oarchive& operator<<(oarchive& a, const std::vector<bool>& vec) {
    a << vec.size();
    for (size_t i = 0; i < vec.size(); ++i) {
      a << bool(vec[i]);
    }
    return a;
  }

  //! Deserializes a vector. \relates iarchive
  template <typename T>
  iarchive& operator>>(iarchive& a, std::vector<T>& vec) {
    vec.clear();
    deserialize_range<T>(a, std::inserter(vec, vec.end()));
    return a;
  }

} // namespace sill

#endif
