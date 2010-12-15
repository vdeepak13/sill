
#ifndef PRL_FACTOR_HPP
#define PRL_FACTOR_HPP

#include <prl/base/string_functions.hpp>
#include <prl/global.hpp>
#include <prl/math/gdl_enum.hpp>
#include <prl/factor/traits.hpp>

#include <prl/macros_def.hpp>

namespace prl {

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
    //! Throws an exception if the operation is not supported
    static void check_supported(op_type op, unsigned supported_ops) {
      if (!(supported_ops & (1 << op)))
        throw std::invalid_argument
          (std::string("Unsupported operation: ") + to_string(op));
    }

    //! Throws an exception if the operation is not as given
    static void check_supported(op_type op, op_type supported_op) {
      if (op != supported_op)
        throw std::invalid_argument
          (std::string("Unsupported operation: ") + to_string(op));
    }
  };

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_FACTOR_HPP


