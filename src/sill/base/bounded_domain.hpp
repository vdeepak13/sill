#ifndef SILL_BOUNDED_DOMAIN_HPP
#define SILL_BOUNDED_DOMAIN_HPP

#include <sill/global.hpp>
#include <sill/serialization/range.hpp>

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace sill {

  /**
   * A domain that can hold up to the given pre-specified number of elements.
   */
  template <typename T, size_t N>
  class bounded_domain {
  public:
    // Range concept
    typedef T value_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    //! Creates an empty domain.
    bounded_domain() {
      std::fill_n(elems_, N, T());
    }

    //! Creates a domain with the given elements.
    bounded_domain(std::initializer_list<T> init) {
      assert(init.size() <= N);
      std::fill(std::copy(init.begin(), init.end(), elems_), elems_ + N, T());
    }

    //! Creates a domain equivalent to the given vector.
    bounded_domain(const std::vector<T>& elems) {
      assert(elems.size() <= N);
      std::fill(std::copy(elems.begin(), elems.end(), elems_), elems_ + N, T());
    }

    /*
    bounded_domain(const bounded_domain& other) = default;

    bounded_domain(bounded_domain&& other) {
      std::swap(elems_, other.elems_);
    }

    bounded_domain& operator=(const bounded_domain& other) = default;
    */

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      serialize_range(ar, begin(), end());
    }

    //! Loads the domain from an archive.
    void load(iarchive& ar) {
      clear();
      deserialize_range<T>(ar, begin());
    }

    //! Swaps the content of two domains.
    friend void swap(bounded_domain& a, bounded_domain& b) {
      std::swap(a.elems_, b.elems_);
    }

    //! Returns true if two domains are equal.
    friend bool operator==(const bounded_domain& a, const bounded_domain& b) {
      return std::equal(a.elems_, a.elems_ + N, b.elems_);
    }

    //! Returns true if the two domains are not equal
    friend bool operator!=(const bounded_domain& a, const bounded_domain& b) {
      return !(a == b);
    }

    //! Returns the size of the domain.
    size_t size() const {
      for (size_t i = 0; i < N; ++i) {
        if (!elems_[i]) { return i; }
      }
      return N;
    }

    //! Returns true if the domain is empty.
    bool empty() const {
      return !elems_[0];
    }

    //! Returns the pointer to the beginning of the domain.
    T* begin() {
      return elems_;
    }

    //! Returns the pointer to the beginning of the domain.
    const T* begin() const {
      return elems_;
    }

    //! Returns the pointer to the one past the end of the domain.
    T* end() {
      return elems_ + size();
    }

    //! Returns the pointer to the one pas the end of the domain.
    const T* end() const {
      return elems_ + size();
    }

    //! Returns the element with the given index (NULL if past the end).
    T& operator[](size_t i) {
      return elems_[i];
    }

    //! Returns the element with the given index (NULL if past the end).
    const T& operator[](size_t i) const {
      return elems_[i];
    }

    //! Returns the number of times element is present in the set.
    size_t count(T x) const {
      return std::count(elems_, elems_ + N, x);
    }

    //! Appends a new element at the end of the domain.
    void push_back(T x) {
      assert(bool(x));
      size_t pos = size();
      assert(pos < N);
      elems_[pos] = x;
    }

    //! Inserts a new element at the end of the domain if it is already present.
    std::pair<T*, bool> insert(T x) {
      assert(bool(x));
      for (size_t i = 0; i < N; ++i) {
        if (elems_[i] == x) {
          return std::make_pair(elems_ + i, false);
        } else if (!elems_[i]) {
          elems_[i] = x;
          return std::make_pair(elems_ + i, true);
        }
      }
      throw std::runtime_error("bounded_domain::insert: the domain is full");
    }

    //! Erases an element from the domain.
    size_t erase(T x) {
      assert(bool(x));
      T* end = std::remove(elems_, elems_ + N, x);
      std::fill(end, elems_ + N, T());
      return elems_ + N - end;
    }

    //! Erases all elements from the domain.
    void clear() {
      std::fill_n(elems_, N, T());
    }

  private:
    T elems_[N];

  }; // class bounded_domain

  /**
   * Prints the domain to an output stream.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  std::ostream& operator<<(std::ostream& out, const bounded_domain<T, N>& a) {
    for (size_t i = 0; i < N; ++i) {
      if (!a[i]) break;
      out << (i == 0 ? "[" : ", ") << a[i];
    }
    out << "]";
    return out;
  }

  /**
   * Returns the union of two bounded domains, preserving the order of elements
   * in the first one.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bounded_domain<T, N>
  left_union(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    bounded_domain<T, N> r(a);
    size_t i = r.size();
    for (T x : b) {
      if (r.count(x) == 0) {
        if (i == N) {
          throw std::runtime_error("bounded_domain left_union: domain is full");
        }
        r[i++] = x;
      }
    }
    return r;
  }

  /**
   * Returns the union of two boudned domains, preserving the order of elements
   * in the second one.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bounded_domain<T, N>
  right_union(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    return left_union(b, a);
  }

  /**
   * Concatenates two domains. The domains should not have any elements in common.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bounded_domain<T, N>
  concat(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    bounded_domain<T, N> r(a);
    size_t na = a.size();
    size_t nb = b.size();
    if (na + nb > N) {
      throw std::runtime_error("bounded_domain concat: domain is full");
    }
    std::copy(b.begin(), b.begin() + nb, r.begin() + na);
    return r;
  }

  /**
   * Returns an ordered intersection of two domains.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bounded_domain<T, N>
  intersection(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    bounded_domain<T, N> r;
    size_t i = 0;
    for (T x : a) {
      if (b.count(x)) {
        r[i++] = x;
      }
    }
    return r;
  }

  /**
   * Returns the difference of two domains.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bounded_domain<T, N>
  difference(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    bounded_domain<T, N> r;
    size_t i = 0;
    for (T x : a) {
      if (!b.count(x)) {
        r[i++] = x;
      }
    }
    return r;
  }

  /**
   * Returns a pair consisting of the intersection and difference of two domains.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  std::pair<bounded_domain<T, N>, bounded_domain<T, N> >
  partition(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    std::pair<bounded_domain<T, N>, bounded_domain<T, N> > r;
    size_t i = 0;
    size_t j = 0;
    for (T x : a) {
      if (b.count(x)) {
        r.first[i++] = x;
      } else {
        r.second[j++] = x;
      }
    }
    return r;
  }

  /**
   * Returns true if two domains are disjoint.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bool disjoint(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    for (T x : a) {
      if (b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bool equivalent(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    if (a.size() != b.size()) { return false; }
    for (T x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are equivalent if
   * they have the same sets of elements, disregarding their order.
   * \relates bounded_domain
   */
  template <typename T>
  bool equivalent(const bounded_domain<T, 2>& a, const bounded_domain<T, 2>& b) {
    return std::minmax(a[0], a[1]) == std::minmax(b[0], b[1]);
  }

  /**
   * Returns true if all the elements of the first domain are also present in
   * the second domain.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bool subset(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    for (T x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the second domain are also present
   * in the first domain.
   * \relates bounded_domain
   */
  template <typename T, size_t N>
  bool superset(const bounded_domain<T, N>& a, const bounded_domain<T, N>& b) {
    for (T x : b) {
      if (!a.count(x)) { return false; }
    }
    return true;
  }

  // deprecated
  template <typename T, size_t N>
  bounded_domain<T, N> make_domain(const bounded_domain<T, N>& a) {
    return a;
  }

} // namespace sill

#endif
