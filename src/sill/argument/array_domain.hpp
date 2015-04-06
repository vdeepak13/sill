#ifndef SILL_ARRAY_DOMAIN_HPP
#define SILL_ARRAY_DOMAIN_HPP

#include <sill/global.hpp>
#include <sill/serialization/array.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <vector>

namespace sill {

  /**
   * A fixed-size domain that holds exactly the given number of elements.
   * This domain type supports all operations of std::array and can be
   * serialized.
   */
  template <typename T, size_t N>
  class array_domain : public std::array<T, N> {
  public:
    //! Default constructor. Creates an uninitialized (invalid) domain.
    array_domain() { }

    //! Creates a domain with the given elements.
    array_domain(std::initializer_list<T> init) {
      assert(init.size() == N);
      std::copy(init.begin(), init.end(), this->begin());
    }

    //! Creates a domain equivalent to the given vector.
    array_domain(const std::vector<T>& elems) {
      assert(elems.size() == N);
      std::copy(elems.begin(), elems.end(), this->begin());
    }

    //! Returns the number of times an argument is present in the domain.
    size_t count(const T& x) const {
      return std::count(this->begin(), this->end(), x);
    }
  }; // class array_domain

  /**
   * Prints the domain to an output stream.
   * \relates array_domain
   */
  template <typename T, size_t N>
  std::ostream& operator<<(std::ostream& out, const array_domain<T, N>& a) {
    for (size_t i = 0; i < N; ++i) {
      if (!a[i]) break;
      out << (i == 0 ? '[' : ',') << a[i];
    }
    out << ']';
    return out;
  }

  // Set operations
  //============================================================================

  /**
   * The concatentation of two fixed-size domains.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  array_domain<T, M+N>
  operator+(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    array_domain<T, M+N> r;
    std::copy(a.begin(), a.end(), r.begin());
    std::copy(b.begin(), b.end(), r.begin() + M);
    return r;
  }

  /**
   * Returns the difference of two fixed-size domains.
   * This operation is valid only if b is a subset of a.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  array_domain<T, M-N>
  operator-(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    static_assert(M > N, "The first argument must be the larger domain");
    array_domain<T, M-N> r;
    size_t i = 0;
    for (T x : a) {
      if (!b.count(x)) {
        assert(i < M-N);
        r[i++] = x;
      }
    }
    assert(i == M-N);
    return r;
  }

  /**
   * Returns the ordered union of two fixed-size domains.
   * This operation is valid only if the two domains are disjoint.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  array_domain<T, M+N>
  operator|(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    assert(disjoint(a, b));
    return a + b;
  }
  
  /**
   * Returns the ordered intersection of two fixed-size domains.
   * This operation is valid only if domain a is a strict subset of b.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  typename std::enable_if<(M < N), array_domain<T, M>>::type
  operator&(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    assert(subset(a, b));
    return a;
  }
  
  /**
   * Returns the ordered intersection of two fixed-size domains.
   * This operation is valid only if domain b is a subset of a.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  typename std::enable_if<(M >= N), array_domain<T, N> >::type
  operator&(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    array_domain<T, N> r;
    size_t i = 0;
    for (T x : a) {
      if (b.count(x)) {
        assert(i < N);
        r[i++] = x;
      }
    }
    assert(i == N);
    return r;
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  bool disjoint(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    for (T x : a) {
      if (b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates array_domain
   */
  template <typename T>
  bool disjoint(const array_domain<T, 1>& a, const array_domain<T, 1>& b) {
    return a[0] != b[0];
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates array_domain
   */
  template <typename T>
  bool disjoint(const array_domain<T, 2>& a, const array_domain<T, 2>& b) {
    return a[0] != b[0] && a[1] != b[0] && a[0] != b[1] && a[1] != b[1];
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename T, size_t N>
  bool equivalent(const array_domain<T, N>& a, const array_domain<T, N>& b) {
    array_domain<T, N> as = a;
    array_domain<T, N> bs = b;
    std::sort(as.begin(), as.end());
    std::sort(bs.begin(), bs.end());
    return as == bs;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  typename std::enable_if<M != N, bool>::type
  equivalent(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    return false;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename T>
  bool equivalent(const array_domain<T, 1>& a, const array_domain<T, 1>& b) {
    return a[0] == b[0];
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates array_domain
   */
  template <typename T>
  bool equivalent(const array_domain<T, 2>& a, const array_domain<T, 2>& b) {
    return std::minmax(a[0], a[1]) == std::minmax(b[0], b[1]);
  }

  /**
   * Returns true if all the elements of the first domain are also present in
   * the second domain.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  bool subset(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    for (T x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the first domain are also present in
   * the second domain.
   * \relates array_domain
   */
  template <typename T>
  bool subset(const array_domain<T, 1>& a, const array_domain<T, 2>& b) {
    return a[0] == b[0] || a[0] == b[1];
  }
  
  /**
   * Returns true if all the elements of the second domain are also present
   * in the first domain.
   * \relates array_domain
   */
  template <typename T, size_t M, size_t N>
  bool superset(const array_domain<T, M>& a, const array_domain<T, N>& b) {
    for (T x : b) {
      if (!a.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the second domain are also present
   * in the first domain.
   * \relates array_domain
   */
  template <typename T>
  bool superset(const array_domain<T, 2>& a, const array_domain<T, 1>& b) {
    return a[0] == b[0] || a[1] == b[0];
  }

  // Argument operations
  //============================================================================

  /**
   * Returns the number of assignments for a collection of finite arguments.
   */
  template <typename Arg, size_t N>
  size_t finite_size(const array_domain<Arg, N>& dom) {
    size_t size = 1;
    for (Arg arg : dom) {
      if (std::numeric_limits<size_t>::max() / arg->size() <= size) {
        throw std::out_of_range("finite_size: possibly overflows size_t");
      }
      size *= arg->size();
    }
    return size;
  }

  /**
   * Returns the vector dimensionality for a collection of vector arguments.
   */
  template <typename Arg, size_t N>
  size_t vector_size(const array_domain<Arg, N>& dom) {
    size_t size = 0;
    for (Arg arg : dom) {
      size += arg->size();
    }
    return size;
  }

  /**
   * Returns true if two domains are type-compatible.
   * \relates domain
   */
  template <typename Arg, size_t N>
  bool type_compatible(const array_domain<Arg, N>& a,
                       const array_domain<Arg, N>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
      if (!a[i]->type_compatible(b[i])) {
        return false;
      }
    }
    return true;
  }

} // namespace sill

#endif
