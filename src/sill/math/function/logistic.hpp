#ifndef SILL_LOGISTIC_FUNCTION_HPP
#define SILL_LOGISTIC_FUNCTION_HPP

#include <functional>
#include <cmath>
#include <iostream>

#include <sill/math/vector.hpp>

namespace sill {

  /**
   * A logistic function parameterized by a vector of coefficients
   * \ingroup math_functions
   */
  class logistic : public std::unary_function<const vec&, double> {
    
  public:
    //! Standard constructor
    logistic(const vec& w, double b = 0) : w(w), b(b) { }

    //! Returns the weight parameters
    const vec& weights() const {
      return w;
    }

    //! Returns the bias
    double bias() const {
      return b;
    }

    //! Evaluates the function on a real input
    double operator()(const vec& x) const {
      using std::exp;
      return 1.0 / (1 + exp(-b - dot(w, x)));
    }

    //! Prints the parameters to an output stream
    friend std::ostream& operator<<(std::ostream& out, const logistic& f) {
      out << f.w << ' ' << f.b;
      return out;
    }

    //! Reads the parameters from an input stream
    friend std::istream& operator>>(std::istream& in, logistic& f) {
      in >> std::ws >> f.w >> std::ws >> f.b;
      return in;
    }

  private:
    //! The weights
    vec w;

    //! The bias
    double b;

  }; // class logistic

} // namespace sill

#endif
