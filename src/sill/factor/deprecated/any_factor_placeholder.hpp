#ifndef SILL_ANY_FACTOR_PLACEHOLDER_HPP
#define SILL_ANY_FACTOR_PLACEHOLDER_HPP

#include <string>

#include <boost/type_traits/is_base_of.hpp>

#include <sill/factor/factor.hpp>
#include <sill/archive/xml_oarchive.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // Forward declarations
  namespace impl { }

  /**
   * An interface that provides type erasure for factors.
   * A class that implements this interface holds a copy of the factor
   * and translates weakly-typed invocations to strongly typed
   * invocations on the held factor.
   *
   * Note: Some functions have a trailing underscore, to allow the 
   *       implementation to rely on argument-dependent lookup.
   */
  struct factor_placeholder {

    // Constructors and copies
    virtual factor* copy() const = 0;
    virtual factor_placeholder* clone() const = 0;
    virtual factor_placeholder* make(const factor& f) const = 0;
    
    // Comparisons
    virtual bool operator==(const factor_placeholder& other) const = 0;
    virtual bool operator<(const factor_placeholder& other) const = 0;

    // Accessors
    virtual const factor& get() const = 0; //!< Returns the held factor
    virtual domain arguments() const = 0;
    virtual double operator()(const assignment& a) const = 0;

    // Factor operations
    virtual factor_placeholder* collapse(const domain&, op_type op) const = 0;
    virtual factor_placeholder* restrict(const assignment& a) const = 0;
    virtual void subst_args(const var_map& map) = 0;
    virtual bool is_normalizable() const = 0;
    virtual double norm_constant() const = 0;
    virtual void normalize() = 0;
    virtual assignment arg_max_() const = 0;
    virtual assignment arg_min_() const = 0;
    virtual void save(xml_oarchive& out) const = 0;
    virtual ~factor_placeholder() {};

  }; // interface factor_placeholder

  /**
   * A polymorphic wrapper for a factor
   * @tparam F a type that satisfies the Factor concept and
   *         is a subclass of the factor class
   */
  template <typename F>
  struct factor_wrapper : public factor_placeholder {
    static_assert((boost::is_base_of<factor, F>::value));

    F f;

    factor_wrapper() { }

    factor_wrapper(const F& f) : f(f) {}

    const factor& get() const {
      return f;
    }
    bool operator==(const factor_placeholder& other) const {
      return f == static_cast<const factor_wrapper&>(other).f;
    }
    bool operator<(const factor_placeholder& other) const {
      using namespace impl;
      return f < static_cast<const factor_wrapper&>(other).f;
    }
    domain arguments() const {
      return f.arguments(); // may involve conversion from factor's domain_type
    }
    factor* copy() const {
      return new F(f);
    }
    factor_placeholder* clone() const {
      return new factor_wrapper(f);
    }
    factor_placeholder* make(const factor& f) const {
      return new factor_wrapper(static_cast<const F&>(f));
    }
    factor_placeholder* collapse(const domain& retain, op_type op) const {
      typename F::domain_type retained = intersect(retain, f.arguments());
      return new factor_wrapper(f.collapse(retained, op));
    }
    factor_placeholder* restrict(const assignment& a) const {
      return new factor_wrapper(f.restrict(a));
    }
    double operator()(const assignment& a) const {
      return f(a);
    }
    void subst_args(const var_map& map) {
      f.subst_args(map);
    }
    bool is_normalizable() const {
      return f.is_normalizable();
    }
    double norm_constant() const {
      return f.norm_constant();
    }
    void normalize() {
      f.normalize();
    }
    assignment arg_max_() const {
      using namespace impl;
      return arg_max(f);
    }
    assignment arg_min_() const {
      using namespace impl;
      return arg_min(f);
    }
    void save(xml_oarchive& out) const {
      using namespace impl;
      out << f;
    }

  }; // class factor_wrapper

  namespace impl {

    //! The default implementation of comparison operator
    //! \relates factor_wrapper
    template <typename F>
    bool operator<(const F& x, const F& y) {
      throw std::invalid_argument
        ("operator< is not supported for " + std::string(typeid(F).name()));
    }

    //! The default implementation of arg_max
    //! \relates factor_wrapper
    template <typename F>
    assignment arg_max(const F& f) {
      throw std::invalid_argument
        ("arg_max is not supported for " + std::string(typeid(F).name()));
    }

    //! The default implementation of arg_max
    //! \relates factor_wrapper
    template <typename F>
    assignment arg_min(const F& f) {
      throw std::invalid_argument
        ("arg_min is not supported for " + std::string(typeid(F).name()));
    }

    //! The default implementation of XML serialization
    //! \relates factor_wrapper
    template <typename F>
    xml_oarchive& operator<<(xml_oarchive& /* out */, const F& f) {
      throw std::invalid_argument
        ("xml i/o is not supported for " + std::string(typeid(F).name()));
    }

  } // namespace impl

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

