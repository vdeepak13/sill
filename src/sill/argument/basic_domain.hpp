#ifndef SILL_BASIC_DOMAIN_HPP
#define SILL_BASIC_DOMAIN_HPP

#include <sill/functional/hash.hpp>
#include <sill/range/iterator_range.hpp>
#include <sill/serialization/serialize.hpp>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <vector>

namespace sill {

  /**
   * A domain that holds the elements in an std::vector.
   */
  template <typename Arg>
  class basic_domain : public std::vector<Arg> {
  public:
    //! Default constructor. Creates an empty domain.
    basic_domain() { }

    //! Constructs a domain with given number of empty arguments.
    explicit basic_domain(size_t n)
      : std::vector<Arg>(n) { }

    //! Creates a domain with the given elements.
    basic_domain(std::initializer_list<Arg> init)
      : std::vector<Arg>(init) { }

    //! Creates a domain from the given argument vector.
    basic_domain(const std::vector<Arg>& elems)
      : std::vector<Arg>(elems) { }

    //! Creates a domain from the given argument array.
    template <size_t N>
    basic_domain(const std::array<Arg, N>& elems)
      : std::vector<Arg>(elems.begin(), elems.end()) { }
    
    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    basic_domain(Iterator begin, Iterator end)
      : std::vector<Arg>(begin, end) { }

    //! Creates a domain from the given iterator range.
    template <typename Iterator>
    explicit basic_domain(const iterator_range<Iterator>& range)
      : std::vector<Arg>(range.begin(), range.end()) { }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      ar.serialize_range(this->begin(), this->end());
    }

    //! Laods the domain from an archive.
    void load(iarchive& ar) {
      this->clear();
      ar.deserialize_range<Arg>(std::back_inserter(*this));
    }

    //! Returns the number of times an argument is present in the domain.
    size_t count(const Arg& x) const {
      return std::count(this->begin(), this->end(), x);
    }

    /**
     * Partitions this domain into those elements that are present in the
     * given map and those that are not.
     */
    template <typename Map>
    void partition(const Map& map,
                   basic_domain& intersect, basic_domain& difference) const {
      for (Arg arg : *this) {
        if (map.count(arg)) {
          intersect.push_back(arg);
        } else {
          difference.push_back(arg);
        }
      }
    }

    //! Substitutes arguments in-place according to a map.
    template <typename Map>
    void subst(const Map& map) {
      for (Arg& arg : *this) {
        arg = map.at(arg); // TODO: check compatibility
      }
    }

    //! Sorts the elements of the domain in place.
    basic_domain& sort() {
      std::sort(this->begin(), this->end());
      return *this;
    }

    /**
     * Removes the duplicate elements from the domain in place.
     * Does not preserve the relative ordere of elements in the domain.
     */
    basic_domain& unique() {
      std::sort(this->begin(), this->end());
      auto new_end = std::unique(this->begin(), this->end());
      this->erase(new_end, this->end());
      return *this;
    }

  };

  /**
   * Prints the domain to an output stream.
   * \relates basic_domain
   */
  template <typename Arg>
  std::ostream& operator<<(std::ostream& out, const basic_domain<Arg>& dom) {
    out << '[';
    for (std::size_t i = 0; i < dom.size(); ++i) {
      if (i > 0) { out << ','; }
      out << dom[i];
    }
    out << ']';
    return out;
  }

  // Set operations
  //============================================================================

  /**
   * The concatenation of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator+(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r;
    r.reserve(a.size() + b.size());
    std::copy(a.begin(), a.end(), std::back_inserter(r));
    std::copy(b.begin(), b.end(), std::back_inserter(r));
    return r;
  }

  /**
   * Returns the difference of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator-(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r;
    for (Arg x : a) {
      if (!b.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns the ordered union of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator|(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r = a;
    for (Arg x : b) {
      if (!a.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns the ordered intersection of two domains.
   * \relates basic_domain
   */
  template <typename Arg>
  basic_domain<Arg>
  operator&(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    basic_domain<Arg> r;
    for (Arg x : a) {
      if (b.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates basic_domain
   */
  template <typename Arg, typename MapOrSet>
  bool disjoint(const basic_domain<Arg>& a, const MapOrSet& b) {
    for (Arg x : a) {
      if (b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are
   * equivalent if they have the same elements, disregarding the order.
   * \relates basic_domain
   */
  template <typename Arg>
  bool equivalent(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (Arg x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the first domain are also
   * present in the second domain.
   * \relates basic_domain
   */
  template <typename Arg, typename MapOrSet>
  bool subset(const basic_domain<Arg>& a, const MapOrSet& b) {
    if (a.size() > b.size()) {
      return false;
    }
    for (Arg x : a) {
      if (!b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if all the elements of the second domain are also
   * present in the first domain.
   * \relates basic_domain
   */
  template <typename Arg>
  bool superset(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    return subset(b, a);
  }

  /**
   * Returns true if domain a is a suffix of domain b.
   * \relates basic_domain
   */
  template <typename Arg>
  bool prefix(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
  }

  /**
   * Returns true if domain a is a suffix of domain b.
   * \relates basic_domain
   */
  template <typename Arg>
  bool suffix(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    return a.size() <= b.size()
      && std::equal(a.begin(), a.end(), b.end() - a.size());
  }

  // Argument operations
  //============================================================================

  /**
   * Returns the number of assignments for a collection of finite arguments.
   */
  template <typename Arg>
  size_t finite_size(const basic_domain<Arg>& dom) {
    size_t size = 1;
    for (Arg arg : dom) {
      if (std::numeric_limits<size_t>::max() / arg.size() <= size) {
        throw std::out_of_range("finite_size: possibly overflows size_t");
      }
      size *= arg.size();
    }
    return size;
  }

  /**
   * Returns the vector dimensionality for a collection of vector arguments.
   */
  template <typename Arg>
  size_t vector_size(const basic_domain<Arg>& dom) {
    size_t size = 0;
    for (Arg arg : dom) {
      size += arg.size();
    }
    return size;
  }

  /**
   * Returns true if two domains are type-compatible.
   * \relates basic_domain
   */
  template <typename Arg>
  bool compatible(const basic_domain<Arg>& a, const basic_domain<Arg>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (!compatible(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

} // namespace sill


namespace std {
  //! \relates basic_domain
  template <typename Arg>
  struct hash<sill::basic_domain<Arg>> {
    typedef sill::basic_domain<Arg> argument_type;
    typedef size_t result_type;
    size_t operator()(const sill::basic_domain<Arg>& dom) const {
      return sill::hash_range(dom.begin(), dom.end());
    }
  };
} // namespace std

#endif