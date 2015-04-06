#ifndef SILL_HYBRID_DOMAIN_HPP
#define SILL_HYBRID_DOMAIN_HPP

#include <sill/argument/domain.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>

namespace sill {
  
  /**
   * A domain that consists of a finite and a vector component.
   */
  class hybrid_domain {
  public:
    //! Default construct. Creates an empty domain.
    hybrid_domain() { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(const domain<finite_variable*>& finite,
                  const domain<vector_variable*>& vector)
      : finite_(finite), vector_(vector) { }

    //! Constructs a hybrid domain with the given finite and vector components.
    hybrid_domain(domain<finite_variable*>&& finite,
                  domain<vector_variable*>&& vector)
      : finite_(std::move(finite)), vector_(std::move(vector)) { }

    //! Saves the domain to an archive.
    void save(oarchive& ar) const {
      finite_.save(ar);
      vector_.save(ar);
    }

    //! Loads the domain from an archive.
    void load(iarchive& ar) {
      finite_.load(ar);
      vector_.load(ar);
    }

    //! Returns the finite component of the domain.
    domain<finite_variable*>& finite() {
      return finite_;
    }

    //! Returns the finite component of the domain.
    const domain<finite_variable*>& finite() const {
      return finite_;
    }

    //! Returns the vector component of the domain.
    domain<vector_variable*>& vector() {
      return vector_;
    }

    //! Returns the vector component of the domain.
    const domain<vector_variable*>& vector() const {
      return vector_;
    }

    //! Returns the number of variables in this domain.
    size_t size() const {
      return finite_.size() + vector_.size();
    }

    //! Returns the number of finite variables in this domain.
    size_t finite_size() const {
      return finite_.size();
    }

    //! Returns the total dimensionality of the vector variables.
    size_t vector_size() const {
      return sill::vector_size(vector_);
    }

    //! Returns true if the domain contains no arguments.
    bool empty() const {
      return finite_.empty() && vector_.empty();
    }

    //! Returns the number of times a variable is present in the domain.
    size_t count(finite_variable* v) const {
      return finite_.count(v);
    }

    //! Returns the number of times a variable is present in the domain.
    size_t count(vector_variable* v) const {
      return vector_.count(v);
    }

    //! Returns the number of times a variable is present in the domain.
    size_t count(variable* v) const {
      switch (v->type()) {
      case variable::FINITE_VARIABLE:
        return finite_.count(dynamic_cast<finite_variable*>(v));
      case variable::VECTOR_VARIABLE:
        return vector_.count(dynamic_cast<vector_variable*>(v));
      default:
        return 0;
      }
    }

    //! Returns true if two hybrid domains have the same variables.
    bool operator==(const hybrid_domain& other) const {
      return finite_ == other.finite_ && vector_ == other.vector_;
    }

    //! Returns true if two hybrid domian do not have the same variables.
    bool operator!=(const hybrid_domain& other) const {
      return !(*this == other);
    }

    /**
     * Partitions this domain into those elements that are present in
     * the given map and those that are not.
     */
    template <typename Map>
    void partition(const Map& map,
                   hybrid_domain& intersect, hybrid_domain& difference) const {
      finite_.partition(map, intersect.finite_, difference.finite_);
      vector_.partition(map, intersect.vector_, difference.vector_);
    }

    // Mutations
    //==========================================================================

    //! Removes all variables from the domain.
    void clear() {
      finite_.clear();
      vector_.clear();
    }

    //! Substitutes arguments in-place according to a map.
    template <typename Map>
    void subst(const Map& map) {
      finite_.subst(map);
      vector_.subst(map);
    }

  private:
    //! The finite component.
    domain<finite_variable*> finite_;
    
    //! The vector component.
    domain<vector_variable*> vector_;

  }; // struct hybrid_domain

  /**
   * Prints the hybrid domain to an output stream.
   * \relates hybrid_domain
   */
  inline std::ostream& operator<<(std::ostream& out, const hybrid_domain& d) {
    out << '(' << d.finite() << ", " << d.vector() << ')';
    return out;
  }

  /**
   * Swaps the contents of two hybrid domains.
   */
  inline void swap(hybrid_domain& a, hybrid_domain& b) {
    swap(a.finite(), b.finite());
    swap(a.vector(), b.vector());
  }

  // Set operations
  //============================================================================

  /**
   * Returns the concatenation of two hybrid domains.
   * \relates hybrid_domain
   */
  inline hybrid_domain
  operator+(const hybrid_domain& a, const hybrid_domain& b) {
    return hybrid_domain(a.finite() + b.finite(), a.vector() + b.vector());
  }

  /**
   * Returns the difference of two hybrid domains.
   * \relates hybrid_domain
   */
  inline hybrid_domain
  operator-(const hybrid_domain& a, const hybrid_domain& b) {
    return hybrid_domain(a.finite() - b.finite(), a.vector() - b.vector());
  }

  /**
   * Returns the ordered union of two hybrid domains.
   * \relates hybrid_domain
   */
  inline hybrid_domain
  operator|(const hybrid_domain& a, const hybrid_domain& b) {
    return hybrid_domain(a.finite() | b.finite(), a.vector() | b.vector());
  }

  /**
   * Returns the ordered intersection of two hybrid domains.
   * \relates hybrid_domain
   */
  inline hybrid_domain
  operator&(const hybrid_domain& a, const hybrid_domain& b) {
    return hybrid_domain(a.finite() & b.finite(), a.vector() & b.vector());
  }

  /**
   * Returns true if two hybrid domains do not have nay elements in common.
   * \relates hybrid_domain
   */
  inline bool disjoint(const hybrid_domain& a, const hybrid_domain& b) {
    return disjoint(a.finite(), b.finite())
      && disjoint(a.vector(), b.vector());
  }

  /**
   * Returns true if two hybrid domains are equivalent.
   * Two hybrid domains are equivalent if their respective finite and
   * vector components are equivalent.
   * \relates hybrid_domain
   */
  inline bool equivalent(const hybrid_domain& a, const hybrid_domain& b) {
    return equivalent(a.finite(), b.finite())
      && equivalent(a.vector(), b.vector());
  }

  /**
   * Returns true if all the elements of the first domain are also
   * present in the second domain.
   * \relates hybrid_domain
   */
  inline bool subset(const hybrid_domain& a, const hybrid_domain& b) {
    return subset(a.finite(), b.finite())
      && subset(a.vector(), b.vector());
  }

  /**
   * Returns true if all the elements of the second domain are also
   * present in the first domain.
   * \relates hybrid_domain
   */
  inline bool superset(const hybrid_domain& a, const hybrid_domain& b) {
    return superset(a.finite(), b.finite())
      && superset(a.vector(), b.vector());
  }
  
  /**
   * Returns true if two domains are type-compatible.
   * \relates hybrid_domain
   */
  inline bool type_compatible(const hybrid_domain& a,
                              const hybrid_domain& b) {
    return type_compatible(a.finite(), b.finite())
      && type_compatible(a.vector(), b.vector());
  }

} // namespace sill

#endif
