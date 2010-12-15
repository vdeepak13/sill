
#ifndef PRL_OPTIMIZATION_CONCEPTS_HPP
#define PRL_OPTIMIZATION_CONCEPTS_HPP

#include <ostream>

#include <prl/global.hpp>
#include <prl/stl_concepts.hpp>

#include <prl/macros_def.hpp>

/**
 * \file concepts.hpp Concepts for convex optimization.
 */

namespace prl {

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

    typedef typename V::size_type size_type;

    // Constructors and destructors
    //--------------------------------------------------------------------------

    // TO DO: ADD 2 CONSTRUCTOR CONCEPTS:
    // void V(size_type s);
    // void V(size_type s, double default_val);

    //! Constructor.
    //    void V(size_type s, double default_val);

    // Getters and non-math setters
    //--------------------------------------------------------------------------

    //! Returns true iff this instance equals the other.
    bool operator==(const V& other) const;

    //! Returns false iff this instance equals the other.
    bool operator!=(const V& other) const;

    //! Returns the dimensions of this data structure.
    size_type size() const;

    //! Resize the data.
    void resize(const size_type& newsize);

    // Math operations
    //--------------------------------------------------------------------------

    //! Sets all elements to this value.
    V& operator=(double d);

    //! Addition.
    V operator+(const V& other) const;

    //! Addition.
    V& operator+=(const V& other);

    //! Subtraction.
    V operator-(const V& other) const;

    //! Subtraction.
    V& operator-=(const V& other);

    //! Multiplication by a scalar value.
    V operator*(double d) const;

    //! Multiplication by a scalar value.
    V& operator*=(double d);

    //! Division by a scalar value.
    V operator/(double d) const;

    //! Division by a scalar value.
    V& operator/=(double d);

    //! Inner product with a value of the same size.
    double inner_prod(const V& other) const;

    //! Element-wise multiplication with another value of the same size.
    V& elem_mult(const V& other);

    //! Element-wise reciprocal (i.e., change v to 1/v).
    V& reciprocal();

    //! Returns the L1 norm.
    double L1norm() const;

    //! Returns the L2 norm.
    double L2norm() const;

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    V sign() const;

    /**
     * "Zeros" this vector.  Normally, this means setting all values to 0.
     * However, for some OptimizationVectors, this could use other natural
     * default values; for example, a matrix which must be positive semidefinite
     * could be set to be the identity matrix.
     */
    void zeros();

    //! Print info about this vector (for debugging).
    void print_info(std::ostream& out) const;

    concept_usage(OptimizationVector) {
      V v2(const_vref.size(), 0.);
      prl::same_type(vref, vref = d);
      prl::same_type(const_vref.size(), s);
      vref.resize(s);
      prl::same_type(const_vref + const_vref, v);
      prl::same_type(v += const_vref, vref);
      prl::same_type(const_vref - const_vref, v);
      prl::same_type(v -= const_vref, vref);
      prl::same_type(const_vref * d, v);
      prl::same_type(v *= d, vref);
      prl::same_type(d, const_vref.inner_prod(const_vref));
      prl::same_type(vref, vref.elem_mult(const_vref));
      prl::same_type(vref, vref.reciprocal());
      prl::same_type(d, const_vref.L1norm());
      prl::same_type(d, const_vref.L2norm());
      prl::same_type(const_vref.sign(), v);
      vref.zeros();
      const_vref.print_info(out);
    }

  private:
    V v;
    static V& vref;
    static const V& const_vref;
    size_type s;
    double d;
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
      prl::same_type(d, f.objective(d));
      prl::same_type(b, f.stop_early());
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
      prl::same_type(d, f.gradient(d));
    }

  private:
    double d;
    static const F& f;

  }; // struct LineSearchGradientFunctor

  /**
   * Concept for a functor which computes an objective at x.
   * @tparam OptVectorType  Type used to store x.
   */
  template <class F, typename OptVectorType>
  struct ObjectiveFunctor {

    //! Computes the value of the objective at x.
    double objective(const OptVectorType& x) const;

    concept_usage(ObjectiveFunctor) {
      prl::same_type(d, f.objective(cvt));
    }

  private:
    double d;
    static const F& f;
    static const OptVectorType& cvt;

  }; // struct ObjectiveFunctor

  /**
   * Concept for a functor which computes the gradient of a function at x.
   * @tparam OptVectorType  Type used to store the gradient and x.
   */
  template <class F, typename OptVectorType>
  struct GradientFunctor {

    //! Computes the gradient of the function at x.
    //! @param grad  Location in which to store the gradient.
    void gradient(OptVectorType& grad, const OptVectorType& x) const;

    concept_usage(GradientFunctor) {
      f.gradient(vt, cvt);
    }

  private:
    static const F& f;
    static OptVectorType& vt;
    static const OptVectorType& cvt;

  }; // struct GradientFunctor

  /**
   * Concept for a functor which applies a preconditioner to a direction
   * vector, when the optimization is from the point x.
   * @tparam OptVectorType  Type used to store x and the direction.
   */
  template <class F, typename OptVectorType>
  struct PreconditionerFunctor {

    //! Applies a preconditioner to the given direction,
    //! when the optimization variables have value x.
    void precondition(OptVectorType& direction, const OptVectorType& x) const;

    //! Applies the last computed preconditioner to the given direction.
    void precondition(OptVectorType& direction) const;

    concept_usage(PreconditionerFunctor) {
      f.precondition(vt, cvt);
      f.precondition(vt);
    }

  private:
    static const F& f;
    static OptVectorType& vt;
    static const OptVectorType& cvt;

  }; // struct PreconditionerFunctor

  /**
   * Concept for a functor which computes the diagonal of a Hessian
   * of a function at x.
   * @tparam OptVectorType  Type used to store the diagonal and x.
   */
  template <class F, typename OptVectorType>
  struct HessianDiagFunctor {

    //! Computes the diagonal of a Hessian of the function at x.
    //! @param hd  Location in which to store the diagonal.
    void hessian_diag(OptVectorType& hd, const OptVectorType& x) const;

    concept_usage(HessianDiagFunctor) {
      f.hessian_diag(vt, cvt);
    }

  private:
    static const F& f;
    static OptVectorType& vt;
    static const OptVectorType& cvt;

  }; // struct HessianDiagFunctor

  /**
   * Concept for a functor which computes the Hessian of a function at x.
   * @tparam OptVectorType  Type used to store x.
   * @tparam HessianType      Type used to store the Hessian.
   */
  template <class F, typename OptVectorType, typename HessianType>
  struct HessianFunctor {

    //! Computes the Hessian of the function at x.
    //! @param h  Location in which to store the Hessian.
    void hessian(HessianType& h, const OptVectorType& x) const;

    concept_usage(HessianFunctor) {
      f.hessian(ht, cvt);
    }

  private:
    static const F& f;
    static HessianType& ht;
    static const OptVectorType& cvt;

  }; // struct HessianFunctor

  //! @} group optimization_concepts

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_OPTIMIZATION_CONCEPTS_HPP
