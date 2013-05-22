#ifndef SILL_SOFT_MAX_HPP
#define SILL_SOFT_MAX_HPP

#include <iosfwd>

#include <sill/base/finite_variable.hpp>
#include <sill/base/assignment.hpp>
#include <sill/math/ublas/special_matrics.hpp>
#include <sill/factor/factor.hpp>

namespace sill {

  // Forward declarations of related factors
  class constant_factor;
  class canonical_gaussian;
  class table_factor;
  
  /**
   * A factor that implements the soft-max distribution.
   */
  class soft_max : public factor {
  
    // Public type declarations
    //==========================================================================
  public:
    //! implements Factor::domain_type
    typedef finite_domain domain_type;

    //! implements Factor::variable_type
    typedef variable variable_type;

    //! implements Factor::collapse_type
    typedef soft_max collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = 0;

    //! implements Factor::combine_ops
    static const unsigned combine_ops = 1 << product_op;

    // Private data members
    //==========================================================================
  private:
    //! The matrix of weights
    mat w;

    //! The vector of biases
    mat b;

    //! The head variable
    finite_variable* head;

    //! The tail variable
    var_vector tail;

    //! The arguments
    domain args;

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Default constructor (invalid distribution with no tail)
    soft_max() : head() { }

    //! Constructs a soft-max CPD with the given head and tail variables
    soft_max(finite_variable* head, const var_vector& tail);

    //! Constructs a soft-max CPDwith the specified parameters
    soft_max(finite_variable* head, const var_vector& tail, 
             const mat& w, const vec& tail);
    
    //! Conversion from constant factor
    soft_max(const constant_factor& other);

    //! Conversion to human-readable representation
    operator std::string() const;

    // Accessors
    //==========================================================================
    //! Returns the arguments of the factor
    const domain& arguments() const;

    //! Returns the tail arguments of the factor
    const var_vector& tail_list() const { return tail; }

    //! Returns the head variable of the factor
    finite_variable* head_var() const { return head; }
    
    // Factor operations
    //==========================================================================
    //! Evaluates the factor for the given tail variables
    vec operator()(const vec& tail);

    //! Collapses the factor (not implemented)
    soft_max collapse(const assignment& a) { assert(false); }

    //! Restricts the factor (not implemented yet)
    soft_max restrict(const assignment& a) { assert(false); }

  }; // class soft_max

  //! Prints the factor to a stream
  //! \relates soft_max
  std::ostream& operator<<(std::ostream& out, const soft_max& factor);
  
  // Factor combinations
  //==========================================================================
  //! Combines a soft-max CPD with a Gaussian distribution
  //! \relates soft_max
  canonical_gaussian combine(const soft_max& f, const canonical_gaussian& cg);

  //! Combines a soft-max CPD with a Gaussian distribution
  //! \relates soft_max
  canonical_gaussian combine(const canonical_gaussian& cg, const soft_max& f);

  // templates for type deduction
  template<> class combine_result<soft_max, canonical_gaussian> {
    typedef canonical_gaussian type;
  };
  
  template<> class combine_result<soft_max, canonical_gaussian> {
    typedef canonical_gaussian type;
  };

  template<> class combine_result<soft_max, table_factor> {
    typedef ;
  };
    
  template<> class combine_result<table_factor, soft_max> {
    typedef ;
  };

} // namespace sill


#endif
