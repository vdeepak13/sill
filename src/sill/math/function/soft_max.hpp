#ifndef SILL_SOFT_MAX_FUNCTION_HPP
#define SILL_SOFT_MAX_FUNCTION_HPP

#include <iostream>
#include <functional>

#include <sill/math/linear_algebra.hpp>

namespace sill {

  /**
   * A soft-max function. This function can represent a conditional 
   * distribution over a finite variable.
   * \ingroup math_functions
   */
  class soft_max : public std::unary_function<const vec&, vec> {
    
  public:
    //! Creates a soft-max function with the given set of weights and biases
    soft_max(const mat& w, const vec& b) : w(w), b(b) {
      assert(w.size1() == b.size());
    }

    //! Returns the weight parameters
    const mat& weights() const {
      return w;
    }

    //! Returns the bias parameters
    const vec& biases() const {
      return b;
    }    

    //! Evaluates the function on a real input
    vec operator()(const vec& x) const {
      vec y = exp(w*x + b);
      return y /= sum(y);
    }

    /*
    //! Evaluates te function on an integer input
    vec operator()(const ivec& x) const {
      vec y = exp(w*to_vec(x) + b);
      return y /= sum(y);
    }
    */

    //! Prints the function parameters to a stream
    friend std::ostream& operator<<(std::ostream& out, const soft_max& f) {
      out << f.w << ' ' << f.b;
      return out;
    }

    //! Load the function parameters from a stream
    friend std::istream& operator>>(std::istream& in, soft_max& f) {
      in >> std::ws >> f.w >> std::ws >> f.b;
      return in;
    }

  private:
    mat w;
    vec b;
    
  }; // class soft_max
  

} // namespace sill

#endif
