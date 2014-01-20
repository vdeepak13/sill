#ifndef SILL_INPLACE_HPP
#define SILL_INPLACE_HPP

// Function objects representing inplace operations
namespace sill {
  
  // base class
  template <typename T>
  struct inplace_op {
    typedef T&       first_argument;
    typedef const T& second_argument;
    typedef T&       result_type;
  };
  
  // inplace addition
  template <typename T>
  struct inplace_plus : public inplace_op<T> {
    T& operator()(T& a, const T& b) {
      return a += b;
    }
    T initial_value() {
      return T(0);
    }
  };

  // inplace subtraction
  template <typename T>
  struct inplace_minus : public inplace_op<T> {
    T& operator()(T& a, const T& b) {
      return a -= b;
    }
    T initial_value() {
      return T(0);
    }
  };

  // inplace multiplication
  template <typename T>
  struct inplace_multiplies : public inplace_op<T> {
    T& operator()(T& a, const T& b) {
      return a *= b;
    }
    T initial_value() {
      return T(1);
    }
  };

  // inplace division
  template <typename T>
  struct inplace_divides : public inplace_op<T> {
    T& operator()(T& a, const T& b) {
      return a /= b;
    }
    T initial_value() {
      return T(1);
    }
  };

} // namespace sill

#endif
