#ifndef SILL_SERIALIZE_ARRAY_HPP
#define SILL_SERIALIZE_ARRAY_HPP

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

#include <array>

namespace sill {

  /**
   * Serializes an std::array.
   * \relates oarchive
   */
  template <typename T, size_t N>
  oarchive& operator<<(oarchive& ar, const std::array<T, N>& a){
    for (size_t i = 0; i < N; ++i) {
      ar << a[i];
    }
    return ar;
  }

  /**
   * Deserializes an std::array.
   * \relates iarchive
   */
  template <typename T, size_t N>
  iarchive& operator>>(iarchive& ar, std::array<T, N>& a) {
    for (size_t i = 0; i < N; ++i) {
      ar >> a[i];
    }
    return ar;
  }

} // namespace sill

#endif
