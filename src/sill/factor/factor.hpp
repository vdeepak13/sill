#ifndef SILL_FACTOR_HPP
#define SILL_FACTOR_HPP

#include <sill/global.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * The base class of all factors.  
   *
   * This class provides basic checking, and is used to enable the
   * standard factor operators (*=, /=, etc.), see is_factor.
   * For a list of functions that are satisfied by all factor
   * types, see the Factor concept.
   *
   * Note that the class has a virtual destructor (through the base
   * class serializable).  Since factors are often allocated on the
   * stack, the vtable lookup is typically optimized away even in the
   * performance-oriented factors, such as
   * fixed_factors. Nevertheless, the factor implementations are free
   * to not subclass from this class and override is_factor.
   *
   * \ingroup factor_types
   */
  class factor {
    
    // Constructors
    // =========================================================================
  protected:
    //! The default constructor with no arguments
    factor() { }

    // Public static functions
    // =========================================================================
  public:
    virtual ~factor() { }
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
