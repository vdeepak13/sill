#ifndef SILL_FUNCTIONAL_ASSIGN_HPP
#define SILL_FUNCTIONAL_ASSIGN_HPP

namespace sill {

  //! Adds one object to another one in place.
  template <typename T = void>
  struct plus_assign {
    auto operator()(T& a, const T& b) const -> decltype(a += b) {
      return a += b;
    }
  };

  //! Adds one object to another one in place.
  template <>
  struct plus_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a += b) {
      return a += b;
    }
  };

  //! Subtracts one object from another one in place.
  template <typename T = void>
  struct minus_assign {
    auto operator()(T& a, const T& b) const -> decltype(a -= b) {
      return a -= b;
    }
  };

  //! Subtracts one object from another one in place.
  template <>
  struct minus_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a -= b) {
      return a -= b;
    }
  };

  //! Multiplies one object by another one in place.
  template <typename T = void>
  struct multiplies_assign {
    auto operator()(T& a, const T& b) const -> decltype(a *= b) {
      return a *= b;
    }
  };

  //! Multiplies one object by another one in place.
  template <>
  struct multiplies_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a *= b) {
      return a *= b;
    }
  };

  //! Divides one object by another one in place.
  template <typename T = void>
  struct divides_assign {
    auto operator()(T& a, const T& b) const -> decltype(a /= b) {
      return a /= b;
    }
  };

  //! Divides one object by another one in place.
  template <>
  struct divides_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a /= b) {
      return a /= b;
    }
  };

  //! Performs the modulus operation in place.
  template <typename T = void>
  struct modulus_assign {
    auto operator()(T& a, const T& b) const -> decltype(a %= b) {
      return a %= b;
    }
  };

  //! Performs the modulus operation in place.
  template <>
  struct modulus_assign<void> {
    template <typename T, typename U>
    auto operator()(T&& a, const U& b) const -> decltype(a %= b) {
      return a %= b;
    }
  };

} // namespace sill

#endif
