#ifndef PRL_FACTOR_APPROXIMATION_INTERFACES_HPP
#define PRL_FACTOR_APPROXIMATION_INTERFACES_HPP

#include <functional>

#include <prl/factor/moment_gaussian.hpp>
#include <prl/factor/nonlinear_gaussian.hpp>

namespace prl {

  // forward declarations
  class moment_gaussian;
  class nonlinear_gaussian;

  /**
   * An interface that can approximate a nonlinear_gaussian.
   * \ingroup factor_approx
   */
  struct gaussian_approximator 
    : public std::binary_function<const nonlinear_gaussian&,
                                  const moment_gaussian&, 
                                  moment_gaussian> {

    //! Approximates the product of ng and prior
    virtual moment_gaussian operator()(const nonlinear_gaussian& ng,
                                       const moment_gaussian& prior) const = 0;

    //! Returns a new dynamically allocated copy of this object
    virtual gaussian_approximator* clone() const = 0;

    //! Destructor
    virtual ~gaussian_approximator() { }

  protected:
    //! Default constructor
    gaussian_approximator() { }

    //! Prevent accidental copies (only the descendants can make copies)
    gaussian_approximator(const gaussian_approximator& other) { }

  }; // interface gaussian_approximator

} // namespace prl

#endif
