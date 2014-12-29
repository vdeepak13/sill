#ifndef SILL_GRADIENT_METHOD_HPP
#define SILL_GRADIENT_METHOD_HPP

#include <sill/optimization/concepts.hpp>
#include <sill/optimization/line_search/line_search.hpp>

#include <iostream>

#include <boost/function.hpp>

namespace sill {
  
  /**
   * An interface for gradient-based optimization algorithms that
   * minimize the given objective.
   *
   * \tparam Vec the type of the optimization vector
   */
  template <typename Vec>
  class gradient_method {
  public:
    //! The storage type of the vector
    typedef typename Vec::value_type real_type;

    //! A type that represents the step and the corresponding objective value
    typedef line_search_result<real_type> result_type;

    //! A type that represents the objective function value
    typedef boost::function<real_type(const Vec&)> objective_fn;

    //! A type that represents the gradient of the objective
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    /**
     * Default constructor.
     */
    gradient_method() { }

    /**
     * Destructor.
     */
    virtual ~gradient_method() { }

    /**
     * Resets the objective, gradient, and the initial estimate.
     */
    virtual void reset(const objective_fn& objective,
                       const gradient_fn& gradient,
                       const Vec& init) = 0;

    /**
     * Performs one iteration.
     * \return the latest line search result (step and objective value)
     */
    virtual result_type iterate() = 0;

    /**
     * Returns true if the iteration has converged.
     */
    virtual bool converged() const = 0;

    /**
     * Returns the solution.
     */
    virtual const Vec& solution() const = 0;

    /**
     * Prints the name of the gradient method and its parameters to an
     * output stream.
     */
    virtual void print(std::ostream& out) const = 0;

  }; // class gradient_method

  /**
   * Prints the gradient_method object to an output stream.
   * \relates gradient_method
   */
  template <typename Vec>
  std::ostream& operator<<(std::ostream& out, const gradient_method<Vec>& gm) {
    gm.print(out);
    return out;
  }
  

} // namespace sill

#endif
