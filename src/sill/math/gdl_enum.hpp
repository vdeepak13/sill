#ifndef SILL_GDL_ENUM_HPP
#define SILL_GDL_ENUM_HPP

#include <iosfwd>
#include <functional>
#include <boost/function.hpp>

#include <sill/global.hpp>
#include <sill/functional.hpp>
#include <sill/range/concepts.hpp>
#include <sill/range/numeric.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  //! A type that describes an operation in a commutative semiring
  //! \ingroup math_functions
  enum op_type { 
    no_op = 0,
    sum_op = 1,
    minus_op = 2,
    product_op = 3,
    divides_op = 4,
    max_op = 5,
    min_op = 6,
    and_op = 7,
    or_op = 8
  };

  //! Prints a string representation of the operation to an output stream
  //! \relates op_type
  inline std::ostream& operator<<(std::ostream& out, op_type op) {
    switch (op) {
    case no_op: out << "(uninitialized)";
    case sum_op: out << "sum";
    case minus_op:  out << "minus";
    case product_op: out << "product";
    case divides_op: out << "divides";
    case max_op: out << "max";
    case min_op: out << "min";
    case and_op: out << "and";
    case or_op: out << "or";
    default: out << "(invalid)";
    }
    return out;
  }

  //! Invokes a function with the functor that corresponds to the op type
  //! \relates op_type
  inline boost::function2<double,double,double> to_functor(op_type op) {
    switch(op) {
    case sum_op: return std::plus<double>();
    case minus_op: return std::minus<double>();
    case product_op: return std::multiplies<double>();
    case divides_op: return std::divides<double>();
    case max_op: return sill::maximum<double>();
    case min_op: return sill::minimum<double>();
    case and_op: return sill::logical_and<double>();
    case or_op: return sill::logical_or<double>();
    default: assert(false); /* this should never be reached.  */ 
             return std::plus<double>(); /* Return to avoid compiler warnings*/
    }
  }
  
  //! Invokes a function with the functor that corresponds to the op type
  //! \relates op_type
  template <typename T, typename UnaryFunction>
  inline typename UnaryFunction::result_type
  apply(UnaryFunction f, op_type op) {
    return f(to_functor(op));
  }
  
/****************** CSR Stuff. To be moved to a different file **************/
  
    /**
   * A type that represents one of pre-defined commutative semirings
   * on numeric datatypes.
   * \ingroup math_functions
   */
  struct commutative_semiring  {
    op_type cross_op;
    op_type dot_op;
    commutative_semiring(op_type cross_op, op_type dot_op)
      : cross_op(cross_op), dot_op(dot_op) { }
  };

  //! Prints a string representation of the commutative semiring
  //! \relates commutative_semiring
  inline std::ostream&
  operator<<(std::ostream& out, const commutative_semiring& csr) {
    out << csr.cross_op << '-' << csr.dot_op;
    return out;
  }

#ifndef SWIG 
  // These constants are defined directly in the SWIG header gdl.i

  //! An object representing the sum product commutative semiring \f$([0,
  //! \infty), +, \times, 0, 1)\f$.
  //! \relates commutative_semiring
  static const commutative_semiring sum_product(sum_op, product_op);

  //! An object representing the max product commutative semiring \f$([0,
  //! \infty), \max, \times, 0, 1)\f$.
  //! \relates commutative_semiring
  static const commutative_semiring max_product(max_op, product_op);

  //! An object representing the min-sum commutative semiring \f$((-\infty,
  //! \infty], \min, +, \infty, 0)\f$.
  //! \relates commutative_semiring
  static const commutative_semiring min_sum(min_op, sum_op);

  //! An object representing the max-sum commutative semiring \f$([-\infty,
  //! \infty), \max, +, -\infty, 0)\f$.
  //! \relates commutative_semiring
  static const commutative_semiring max_sum(max_op, sum_op);

  //! An object representing the Boolean commutative semiring \f$(\{0, 1\},
  //! \lor, \land, 0, 1)\f$.
  //! \relates commutative_semiring
  static const commutative_semiring boolean(or_op, and_op);

#endif

} // namespace sill

#include <sill/macros_undef.hpp>

#endif 
