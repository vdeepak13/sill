#ifndef SILL_HYBRID_DOMAIN_HPP
#define SILL_HYBRID_DOMAIN_HPP

#include <sill/argument/domain.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>

namespace sill {
  
  /**
   * A domain that consists of a finite and vector component.
   */
  template <typename Arg = void>
  struct hybrid_domain {
    domain<finite_variable*> finite;
    domain<vector_variable*> vector;

    hybrid_domain() { }

    hybrid_domain(const domain<finite_variable*>& finite,
                  const domain<vector_varaible*>& vector)
      : finite(finite), vector(vector) { }

    hybrid_domain(domain<finite_variable*>&& finite,
                  domain<vector_variable*>&& vector)
      : finite(std::move(finite)), vector(std::move(vector)) { }

    friend void swap(hybrid_domain& a, hybrid_domain& b) {
      using std::swap;
      swap(a.finite, b.finite);
      swap(a.vector, b.vector);
    }

  }; // struct hybrid_domain

  /**
   * Prints the hybrid domain to an output stream.
   * \relates hybrid_domain
   */
  template <typename Arg>
  std::ostream&
  operator<<(st::ostream& out, const hybrid_domain<Arg>& d) {
    out << '(' << d.finite << ", " << d.vector << ')';
  }

  /**
   * Returns the concatenation of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator+(const hybrid_domain<Arg>& a, const hybrid_domian<Arg>& b) {
    return hybrid_domain<Arg>(a.finite + b.finite, a.vector + b.vector);
  }

  /**
   * Returns the difference of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator-(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.finite - b.finite, a.vector - b.vector);
  }

  /**
   * Returns the ordered union of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator|(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.finite | b.finite, a.vector | b.vector);
  }

  /**
   * Returns the ordered intersection of two hybrid domains.
   * \relates hybrid_domain
   */
  template <typename Arg>
  hybrid_domain<Arg>
  operator&(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return hybrid_domain<Arg>(a.finite & b.finite, a.vector & b.vector);
  }

  /**
   * Returns true if two hybrid domains do not have nay elements in common.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool disjoint(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return disjoint(a.finite, b.finite) && disjoint(a.vector, b.vector);
  }

  /**
   * Returns true if two hybrid domains are equivalent.
   * Two hybrid domains are equivalent if their respective finite and
   * vector components are equivalent.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool equivalent(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return equivalent(a.finite, b.finite) && equivalent(a.vector, b.vector);
  }

  /**
   * Returns true if all the elements of the first domain are also
   * present in the second domain.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool subset(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return subset(a.finite, b.finite) && subset(a.vector, b.vector);
  }

  /**
   * Returns true if all the elements of the second domain are also
   * present in the first domain.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool superset(const hybrid_domain<Arg>& a, const hybrid_domain<Arg>& b) {
    return superset(a.finite, b.finite) && superset(a.vector, b.vector);
  }
  
  /**
   * Returns true if two domains are type-compatible.
   * \relates hybrid_domain
   */
  template <typename Arg>
  bool type_compatible(const hybrid_domain<Arg>& a,
                       const hybrid_domain<Arg>& b) {
    return type_compatible(a.finite, b.finite)
      && type_compatible(a.vector, b.vector);
  }

} // namespace sill

#endif
