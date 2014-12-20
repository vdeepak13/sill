#ifndef SILL_GRADIENT_METHOD_HPP
#define SILL_GRADIENT_METHOD_HPP

#include <sill/optimization/concepts.hpp>

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
    typedef typename Vec::value_type real_type;

    //! Default constructor
    gradient_method() { }

    //! Destructor
    virtual ~gradient_method() { }

    /**
     * Performs one step of iteration.
     * \return true if converged
     */
    virtual bool iterate() = 0;

    //! Returns the current estimate
    virtual const Vec& x() const = 0;

    //! Returns the current objective value
    virtual real_type objective() const = 0;

  }; // class gradient_method

} // namespace sill

#endif
