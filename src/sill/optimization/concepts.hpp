#ifndef SILL_OPTIMIZATION_CONCEPTS_HPP
#define SILL_OPTIMIZATION_CONCEPTS_HPP

#include <ostream>

#include <sill/global.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

/**
 * \file concepts.hpp Concepts for convex optimization.
 */

namespace sill {

  //! \addtogroup optimization_concepts
  //! @{

  /**
   * Concept for a data structure for the variables being optimized over;
   * the data structure must essentially behave like a vector.
   * For many apps, the variables might be stored more naturally in a
   * collection of matrices and vectors, but this abstraction allows
   * optimization algorithms to treat the variables as a vector object.
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

  /**
   * Concept for a functor which computes an objective for line search
   * for different step sizes.
   */
  template <class F>
  struct LineSearchObjectiveFunctor {

    //! Computes the value of the objective for step size eta.
    double objective(double eta) const;

    //! Returns true if this functor recommends early stopping.
    bool stop_early() const;

    concept_usage(LineSearchObjectiveFunctor) {
      sill::same_type(d, f.objective(d));
      sill::same_type(b, f.stop_early());
    }

  private:
    double d;
    static const F& f;
    bool b;

  }; // struct LineSearchObjectiveFunctor

  /**
   * Concept for a functor which computes a gradient
   * for line search for different step sizes.
   */
  template <class F>
  struct LineSearchGradientFunctor {

    //! Computes the gradient of the objective (w.r.t. eta) for step size eta.
    double gradient(double eta) const;

    concept_usage(LineSearchGradientFunctor) {
      sill::same_type(d, f.gradient(d));
    }

  private:
    double d;
    static const F& f;

  }; // struct LineSearchGradientFunctor

  //! @} group optimization_concepts

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
