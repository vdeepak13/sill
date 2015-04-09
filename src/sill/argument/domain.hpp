#ifndef SILL_DOMAIN_HPP
#define SILL_DOMAIN_HPP

#include <sill/serialization/serialize.hpp>

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <vector>

namespace sill {

  /**
   * A domain that holds the elements in an std::vector.
   */
  template <typename Arg>
  class domain : public std::vector<Arg> {
  public:
    //! Default constructor. Creates an empty domain.
    domain() { }

    //! Constructs a domain with given number of empty arguments.
    explicit domain(size_t n)
      : std::vector<Arg>(n) { }

    //! Creates a domain with the given elements.
    domain(std::initializer_list<Arg> init)
      : std::vector<Arg>(init) { }

    //! Creates a from the given argument vector.
    domain(const std::vector<Arg>& elems)
      : std::vector<Arg>(elems) { }

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
                   domain& intersect, domain& difference) const {
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

    /**
     * Removes the duplicate elements from the domain in place.
     * Does not preserve the relative ordere of elements in the domain.
     */
    domain& unique() {
      std::sort(this->begin(), this->end());
      auto new_end = std::unique(this->begin(), this->end());
      this->erase(new_end, this->end());
      return *this;
    }

  };

  /**
   * Prints the domain to an output stream.
   * \relates domain
   */
  template <typename Arg>
  std::ostream& operator<<(std::ostream& out, const domain<Arg>& dom) {
    print_range(out, dom, '[', ',', ']');
    return out;
  }

  // Set operations
  //============================================================================

  /**
   * The concatenation of two domains.
   * \relates domain
   */
  template <typename Arg>
  domain<Arg> operator+(const domain<Arg>& a, const domain<Arg>& b) {
    domain<Arg> r;
    r.reserve(a.size() + b.size());
    std::copy(a.begin(), a.end(), std::back_inserter(r));
    std::copy(b.begin(), b.end(), std::back_inserter(r));
    return r;
  }

  /**
   * Returns the difference of two domains.
   * \relates domain
   */
  template <typename Arg>
  domain<Arg> operator-(const domain<Arg>& a, const domain<Arg>& b) {
    domain<Arg> r;
    for (Arg x : a) {
      if (!b.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns the ordered union of two domains.
   * \relates domain
   */
  template <typename Arg>
  domain<Arg> operator|(const domain<Arg>& a, const domain<Arg>& b) {
    domain<Arg> r = a;
    for (Arg x : b) {
      if (!a.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns the ordered intersection of two domains.
   * \relates domain
   */
  template <typename Arg>
  domain<Arg> operator&(const domain<Arg>& a, const domain<Arg>& b) {
    domain<Arg> r;
    for (Arg x : a) {
      if (b.count(x)) {
        r.push_back(x);
      }
    }
    return r;
  }

  /**
   * Returns true if two domains do not have any elements in common.
   * \relates domain
   */
  template <typename Arg, typename MapOrSet>
  bool disjoint(const domain<Arg>& a, const MapOrSet& b) {
    for (Arg x : a) {
      if (b.count(x)) { return false; }
    }
    return true;
  }

  /**
   * Returns true if two domains are equivalent. Two domains are
   * equivalent if they have the same elements, disregarding the order.
   * \relates domain
   */
  template <typename Arg>
  bool equivalent(const domain<Arg>& a, const domain<Arg>& b) {
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
   * \relates domain
   */
  template <typename Arg, typename MapOrSet>
  bool subset(const domain<Arg>& a, const MapOrSet& b) {
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
   * \relates domain
   */
  template <typename Arg>
  bool superset(const domain<Arg>& a, const domain<Arg>& b) {
    return subset(b, a);
  }

  /**
   * Returns true if domain a is a suffix of domain b.
   * \relates domain
   */
  template <typename Arg>
  bool prefix(const domain<Arg>& a, const domain<Arg>& b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
  }

  /**
   * Returns true if domain a is a suffix of domain b.
   * \relates domain
   */
  template <typename Arg>
  bool suffix(const domain<Arg>& a, const domain<Arg>& b) {
    return a.size() <= b.size()
      && std::equal(a.begin(), a.end(), b.end() - a.size());
  }

  // Argument operations
  //============================================================================

  /**
   * Returns the number of assignments for a collection of finite arguments.
   */
  template <typename Arg>
  size_t finite_size(const domain<Arg>& dom) {
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
  template <typename Arg>
  size_t vector_size(const domain<Arg>& dom) {
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
  template <typename Arg>
  bool type_compatible(const domain<Arg>& a, const domain<Arg>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (!a[i]->type_compatible(b[i])) {
        return false;
      }
    }
    return true;
  }

} // namespace sill

#endif
