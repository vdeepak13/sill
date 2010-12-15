#ifndef PRL_FUNCTION_INTERFACES_HPP
#define PRL_FUNCTION_INTERFACES_HPP

#include <functional>
#include <string>

#include <prl/math/vector.hpp>
#include <prl/math/matrix.hpp>

namespace prl {

  /**
   * A vector-valued function over a vector domain.
   * \ingroup math_functions
   */
  struct vector_function : public std::unary_function<const vec&, vec> {

    //! Conversion to human-readable representation of the function
    virtual operator std::string() const = 0;
    
    //! Returns a new copy of the function, allocated on the heap
    virtual vector_function* clone() const = 0;

    virtual ~vector_function() { }

    //! Evaluates the function on the specified input
    vec operator()(const vec& input) const {
      vec output;
      value(input, output);
      return output;
    }

    //! Evaluates the function and stores the result to the output argument
    virtual void value(const vec& input, vec& output) const = 0;

    //! Returns the dimensionality of the output argument
    virtual size_t size_out() const = 0;

    //! Returns the dimensionality of the input argument
    virtual size_t size_in() const = 0;
    
  }; // class vector_function

  /**
   * A real function over a real vector domain.
   * \ingroup math_functions
   */
  struct real_function : public std::unary_function<const vec&, double> {

    virtual ~real_function() { }
    
    //! Conversion to human-readable representation of the function
    virtual operator std::string() const = 0;

    //! Returns a new copy of the function, allocated on the heap
    virtual real_function* clone() const = 0;
    
    //! Returns the value of the function for the given input
    virtual double operator()(const vec& input) const = 0;

    //! Returns the gradient of the function at the given input
    virtual vec gradient(const vec& input) const = 0;

    //! Returns the Hessian of the function at the given input
    virtual mat hessian(const vec& input) const = 0;

    //! Returns the dimensionality of the input argument
    virtual size_t size() const = 0;

  }; // class real_function

} // namespace prl

#endif
