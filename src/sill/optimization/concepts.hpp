#ifndef SILL_OPTIMIZATION_CONCEPTS_HPP
#define SILL_OPTIMIZATION_CONCEPTS_HPP

#include <ostream>

#include <sill/global.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Concept for a data structure for the variables being optimized over;
   * the data structure must essentially behave like a vector.
   * For many apps, the variables might be stored more naturally in a
   * collection of matrices and vectors, but this abstraction allows
   * optimization algorithms to treat the variables as a vector object.
   *
   * \ingroup optimization_concepts
   */
  template <typename V>
  struct OptimizationVector
    : DefaultConstructible<V>, CopyConstructible<V>, Assignable<V> {

    // Types and data
    //--------------------------------------------------------------------------

    typedef typename V::value_type value_type;

    // Vector operations
    //--------------------------------------------------------------------------

    //! Sets all elements to this value.
    //V& operator=(value_type d);

    //! Addition.
    V operator+(const V& other) const;

    //! Addition.
    V& operator+=(const V& other);

    //! Subtraction.
    V operator-(const V& other) const;

    //! Subtraction.
    V& operator-=(const V& other);

    //! Multiplication by a scalar value.
    V operator*(value_type d) const;

    //! Multiplication by a scalar value.
    V& operator*=(value_type d);

    //! Division by a scalar value.
    V operator/(value_type d) const;

    //! Division by a scalar value.
    V& operator/=(value_type d);

    //! Inner product with a value of the same size.
    friend value_type dot(const V& a, const V& b);

    //! L1 norm of the vector.
    friend value_type norm_1(const V& x);

    //! L2 norm of the vector. Must be eqivalent to sqrt(dot(x, x))
    friend value_type norm_2(const V& x);

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    friend V sign(const V& x);

    //! Prints out the vector to an output stream
    friend std::ostream& operator<<(std::ostream& out, const V& x);

    concept_usage(OptimizationVector) {
      sill::same_type(vcref + vcref, v);
      sill::same_type(v += vcref, vref);
      sill::same_type(vcref - vcref, v);
      sill::same_type(v -= vcref, vref);
      sill::same_type(vcref * val, v);
      sill::same_type(v *= val, vref);
      sill::same_type(vcref / val, v);
      sill::same_type(v /= val, vref);
      sill::same_type(dot(vcref, vcref), val);
      sill::same_type(norm_1(vcref), val);
      sill::same_type(norm_2(vcref), val);
      sill::same_type(sign(vcref), v);
      sill::same_type(out, out << vcref);
    }

  private:
    V v;
    static V& vref;
    static const V& vcref;
    value_type val;
    static std::ostream& out;

  }; // struct OptimizationVector

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
