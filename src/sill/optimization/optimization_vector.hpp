#ifndef SILL_OPTIMIZATION_VECTOR_HPP
#define SILL_OPTIMIZATION_VECTOR_HPP

#include <ostream>

#include <sill/global.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! y += a * x
  template <typename OV>
  void ov_axpy(double a, const OV& x, OV& y) {
    y += x * a;
  }

  /**
   * Interface for a struct holding optimization variables.
   * This struct must behave like a vector by supporting certain operations.
   *
   * For many problems, the variables might be stored more naturally in a
   * collection of matrices and vectors, but this abstraction allows
   * optimization algorithms to treat the variables as a single vector object.
   *
   * @tparam LA  Linear algebra type specifier
   */
  /*
  template <typename LA>
  struct optimization_vector {

    // Public types
    //==========================================================================

    typedef LA la_type;
    typedef typename LA::value_type  value_type;
    typedef typename LA::size_type   size_type;

    // Constructors
    //==========================================================================

    // Instances of this interface should support:
    // void optimization_vector(size_type s);
    // void optimization_vector(size_type s, value_type default_val);

    // Getters and non-math setters
    //==========================================================================

    //! Returns true iff this instance equals the other.
    virtual bool operator==(const optimization_vector& other) const = 0;

    //! Returns false iff this instance equals the other.
    virtual bool operator!=(const optimization_vector& other) const = 0;

    //! Returns the dimensions of this data structure.
    virtual size_type size() const = 0;

    //! Resize the data.
    // TO DO: ADD THIS BACK IN ONCE ARMADILLO TYPES SUPPORT IT
//    void resize(const size_type& newsize);

    // Math operations
    //--------------------------------------------------------------------------

    //! Sets all elements to this value.
    virtual optimization_vector& operator=(value_type d) = 0;

    //! Addition.
    virtual
    optimization_vector operator+(const optimization_vector& other) const = 0;

    //! Addition.
    virtual
    optimization_vector& operator+=(const optimization_vector& other) = 0;

    //! Subtraction.
    virtual
    optimization_vector operator-(const optimization_vector& other) const = 0;

    //! Subtraction.
    virtual optimization_vector& operator-=(const optimization_vector& other)=0;

    //! Multiplication by a scalar value.
    virtual optimization_vector operator*(value_type d) const = 0;

    //! Multiplication by a scalar value.
    virtual optimization_vector& operator*=(value_type d) = 0;

    //! Division by a scalar value.
    virtual optimization_vector operator/(value_type d) const = 0;

    //! Division by a scalar value.
    virtual optimization_vector& operator/=(value_type d) = 0;

    //! Inner product with a value of the same size.
    virtual value_type dot(const optimization_vector& other) const = 0;

    //! Element-wise multiplication with another value of the same size.
    virtual optimization_vector& elem_mult(const optimization_vector& other) =0;

    //! Element-wise reciprocal (i.e., change v to 1/v).
    virtual optimization_vector& reciprocal() = 0;

    //! Returns the L1 norm.
    virtual value_type L1norm() const = 0;

    //! Returns the L2 norm.
    virtual value_type L2norm() const = 0;

    //! Returns a struct of the same size but with values replaced by their
    //! signs (-1 for negative, 0 for 0, 1 for positive).
    virtual optimization_vector sign() const = 0;

    //! "Zeros" this vector.  Normally, this means setting all values to 0.
    //! However, for some types, this could use other natural default values;
    //! for example, a matrix which must be positive semidefinite
    //! could be set to an identity matrix.
    virtual void zeros() = 0;

    //! Print info about this vector (for debugging).
    virtual void print_info(std::ostream& out) const = 0;

  }; // struct optimization_vector
  */

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_OPTIMIZATION_VECTOR_HPP
