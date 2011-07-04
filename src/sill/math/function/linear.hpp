#ifndef SILL_MATH_LINEAR_FUNCTION_HPP
#define SILL_MATH_LINEAR_FUNCTION_HPP

#include <sill/math/function/interfaces.hpp>
#include <sill/math/matrix.hpp>
#include <sill/math/vector.hpp>
#include <sill/math/linear_algebra.hpp>

namespace sill {
  
  /**
   * A linear vector function, i.e., a function of the form f(x) = Ax + b
   * \ingroup math_functions
   */
  class linear_vec : public vector_function {

  private: 
    //! The multiplier
    mat a;

    //! The additive constant
    vec b;

  public:
    linear_vec(const mat& a, const vec& b)
      : a(a), b(b) {
      assert(a.n_rows == b.size());
    }

    operator std::string() const {
      return "linear_vec";
    }

    linear_vec* clone() const {
      return new linear_vec(*this);
    }
    
    void value(const vec& x, vec& y) const {
      assert(x.size() == a.n_cols);
      y = a * x + b;
    }

    size_t size_out() const {
      return a.n_rows;
    }
    
    size_t size_in() const {
      return a.n_cols;
    }
    
  }; // class linear_vec


  /**
   * A linear real-valued function, i.e., a function of the form f(x) = a^Tx+b.
   * \ingroup math_functions
   */
  class linear_real : public real_function {
    
  private: 
    //! The multiplier
    vec a;

    //! The additive constant
    double b;

  public:
    linear_real(const vec& a, double b) : a(a), b(b) { }

    //! Conversion to human-readable representation of the function
    operator std::string() const {
      return "linear_real";
    }

    //! Returns a new copy of the function, allocated on the heap
    real_function* clone() const {
      return new linear_real(*this);
    }
    
    //! Returns the value of the function for the given input
    double operator()(const vec& x) const {
      assert(a.size() == x.size());
      return dot(a, x) + b;
    }

    //! Returns the gradient of the function at the given input
    vec gradient(const vec& x) const {
      assert(x.size() == a.size());
      return a;
    }

    //! Returns the Hessian of the function at the given input
    mat hessian(const vec& x) const {
      assert(x.size() == a.size());
      return zeros(x.size(), x.size());
    }

    //! Returns the dimensionality of the input argument
    size_t size() const {
      return a.size();
    }

  }; // class linear_real

} // namespace sill

#endif
