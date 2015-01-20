#ifndef SILL_FUNCTIONAL_ASSIGN_HPP
#define SILL_FUNCTIONAL_ASSIGN_HPP

namespace sill {

  //! Adds one object to another one in place.
  template <typename T, typename U = T>
  struct plus_assign {
    T& operator()(T& a, const U& b) const { return a += b; }
  };

  //! Subtracts one object from another one in place.
  template <typename T, typename U = T>
  struct minus_assign {
    T& operator()(T& a, const U& b) const { return a -= b; }
  };

  //! Multiplies one object by another one in place.
  template <typename T, typename U = T>
  struct multiplies_assign {
    T& operator()(T& a, const U& b) const { return a *= b; }
  };

  //! Divides one object by anotehr one in place.
  template <typename T, typename U = T>
  struct divides_assign {
    T& operator()(T& a, const U& b) const { return a /= b; }
  };

} // namespace sill

#endif
