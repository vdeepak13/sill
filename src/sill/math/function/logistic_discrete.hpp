#ifndef SILL_LOGISTIC_DISCRETE_FUNCTION_HPP
#define SILL_LOGISTIC_DISCRETE_FUNCTION_HPP

#include <functional>
#include <iosfwd>

#include <sill/math/matrix.hpp>
#include <sill/math/vector.hpp>

namespace sill {

  /**
   * A logistic function that takes a vector of discrete values as an input.
   * The function is parameterized by a weight matrix \f$w_{i,j}\f$ and bias
   * \f$b\f$. Given a vector of discrete values \f$v\f$, the function returns
   * \f$\sigma(b + \sum_i w_{i,v_i}\f$.
   *
   * \ingroup math_functions
   */
  class logistic_discrete : public std::unary_function<const uvec&, double> {

  private:
    //! The weights
    mat w;

    //! The bias
    double b;

  public:
    //! Standard constructor
    logistic_discrete(const mat& w, double b = 0) : w(w), b(b) { }

    //! Returns the weight parameters
    const mat& weights() const {
      return w;
    }

    //! Returns the bias
    double bias() const {
      return b;
    }

    //! Evaluates the function on a discrete input
    double operator()(const uvec& x) const;

    //! Evaluates the function on a discrete input, 
    //! where each input variable \c i is weighted by \c u[i]
    double operator()(const uvec& x, const vec& u) const;

    //! Evaluates the function on a weighted matrix
    double operator()(const mat& x) const;

    //! Prints the parameters to an output stream
    friend std::ostream& operator<<(std::ostream& out, 
                                    const logistic_discrete& f);

    //! Reads the parameters from an input stream
    friend std::istream& operator>>(std::istream& in, logistic_discrete& f);

  }; // class logistic

} // namespace sill

#endif
