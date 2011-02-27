
#ifndef SILL_CONSTANT_FACTOR_HPP
#define SILL_CONSTANT_FACTOR_HPP

#include <sstream>

#include <set>

#include <sill/base/finite_assignment.hpp>
#include <sill/factor/factor.hpp>
#include <sill/global.hpp>
#include <sill/learning/dataset/finite_record.hpp>
#include <sill/math/is_finite.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  /*************************************************************************//**
   * A constant factor represents a constant, i.e., a function of no
   * variables.
   *
   * \ingroup factor_types
   * \see Factor
   ****************************************************************************/
  class constant_factor : public factor {

    // Public type declarations
    // =========================================================================
  public:

    //! The type of the value taken on by this factor
    typedef double result_type;

    //! implements Factor::variable_type (arbitrarily chosen to be finite)
    typedef finite_variable variable_type;

    //! implements Factor::domain_type (arbitrarily chosen to be finite)
    typedef finite_domain domain_type;

    //! implements Factor::assignment_type
    typedef finite_assignment assignment_type;

    //! implements Factor::record_type
    typedef finite_record record_type;

    //! The result of a collapse operation
    typedef constant_factor collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = ~0;  // all (bitwise complement of 0)

    //! implements Factor::combine_ops
    static const unsigned combine_ops = ~0; // all (bitwise complement of 0)

  private:
    domain_type empty_domain;
    // Public data members
    // =========================================================================
  public:
    //! The value taken on by the factor.
    double value;
    
    void serialize(oarchive & ar) const {
      ar << value;
    }

    void deserialize(iarchive& ar) {
      ar >> value;
    }

    // Constructors and conversion operators
    // =========================================================================
  public:
    //! Default constructor for a factor with no arguments, i.e., a constant.
    constant_factor(double value = double()) : value(value) { }

    //! Conversion operator to the numeric type
    operator double() const {
      return value;
    }

    //! Conversion to human-readable format
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    // Accessors and comparison operators
    // =========================================================================

    //! Returns the arguments of the factor (i.e., the empty set)
    const domain_type& arguments() const {
      return empty_domain;
    }

    //! Returns true if the two factors represent the same value
    bool operator==(const constant_factor& other) const {
      return value == other.value;
    }

    //! Returns true if the two factors do not represent the same value
    bool operator!=(const constant_factor& other) const {
      return value != other.value;
    }

    //! Returns true if this factor precedes the other one
    bool operator<(const constant_factor& other) const {
      return value < other.value;
    }

    //! Evaluates this factor for the given assignment
    double operator()(const finite_assignment&) const {
      return value;
    }

    //! Mutable access to the factor value
    double& operator()(const finite_assignment&) {
      return value;
    }

    // Factor operations
    // =========================================================================

    //! implements Factor::combine_in
    constant_factor& combine_in(const constant_factor& y, op_type op) {
      value = to_functor(op).operator()(value, y.value);
      return *this;
    }

    //! implements Factor::collapse
    constant_factor collapse(op_type op, const domain_type&) const {
      return *this;
    }

    //! implements Factor::restrict
    constant_factor restrict(const finite_assignment&) const {
      return *this;
    }

    //! implements Factor::subst_args
    constant_factor& subst_args(const finite_var_map&) {
      return *this;
    }

    //! implements DistributionFactor::is_normalizable
    bool is_normalizable() const { 
      return is_positive_finite(value);
    }

    //! implements DistributionFactor::norm_constant
    double norm_constant() const {
      return value;
    }

    //! implements DistributionFactor::normalize
    constant_factor& normalize() { 
      assert(is_normalizable());
      value = 1;
      return *this;
    }

    //! implements DistributionFactor::marginal
    constant_factor marginal(const domain_type& retain) const {
      return *this;
    }

    //! Computes the maximum for each assignment to the given variables
    constant_factor maximum(const domain_type& retain) const { 
      return *this; 
    }

    //! Computes the minimum for each assignment to the given variables
    constant_factor minimum(const domain_type& retain) const { 
      return *this; 
    }

    //! Returns an assignment that achieves the maximum value
    finite_assignment arg_max() const {
      return finite_assignment();
    }
    
    //! Returns an assignment that achieves the minimum value
    finite_assignment arg_min() const {
      return finite_assignment();
    }
    
  }; // class constant_factor

  //! Writes a human-readable representation of a factor expression
  //! \relates constant_factor
  inline std::ostream& operator<<(std::ostream& out, const constant_factor& f) {
    out << "#F(C|" << f.value << ")";
    return out;
  }

  // Result of combining constant factor and any other factor
  template<> struct combine_result<constant_factor, constant_factor> {
    typedef constant_factor type;
  };

  template <typename F>
  struct combine_result< constant_factor, F > {
    typedef F type;
  };

  template <typename F>
  struct combine_result< F, constant_factor > {
    typedef F type;
  };

  /**
   * Combines two constant factors. This template needs to be defined,
   * because the default template in factor.hpp is overshadowed by
   * the two templates below, and the templates are ambiguous for
   * a pair of constant factors.
   * \relates constant_factor
   */
  inline constant_factor
  combine(constant_factor c1, const constant_factor& c2, op_type op) {
    // Note that c1 is passed by value, hence we're modifying a copy
    return c1.combine_in(c2, op);
  }

  //! Combines constant factor and an arbitrary factor
  //! \relates constant_factor
  template <typename F>
  F combine(const constant_factor& c, const F& f, op_type op) {
    return F(c).combine_in(f, op);
  }

  //! Combines constant factor and an arbitrary factor
  //! \relates constant_factor
  template <typename F>
  F combine(F f, const constant_factor& c, op_type op) {
    return f.combine_in(c, op);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CONSTANT_FACTOR_HPP
