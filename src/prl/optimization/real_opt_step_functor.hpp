
#ifndef SILL_REAL_OPT_STEP_FUNCTOR_HPP
#define SILL_REAL_OPT_STEP_FUNCTOR_HPP

#include <limits>

//#include <sill/optimization/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Interface for functors used by real_opt_step to translate step
   * sizes eta into objectives.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_classes
   */
  class real_opt_step_functor {

    // Public methods
    //==========================================================================
  public:

    real_opt_step_functor() { }

    virtual ~real_opt_step_functor() { }

    //! Computes the value of the objective for step size eta.
    virtual double objective(double eta) const = 0;

    //! Returns true if the last call to objective() or gradient() recommended
    //! early stopping (for line search).
    virtual bool stop_early() const {
      return false;
    }

    //! Computes the gradient of the objective (w.r.t. eta) for step size eta.
    //! NOTE: This may not be implemented; use valid_gradient() to check.
    virtual double gradient(double eta) const {
      assert(false);
      return std::numeric_limits<double>::infinity();
    }

    //! Returns true if the gradient() method has been implemented.
    virtual bool valid_gradient() const {
      return false;
    }

  };  // class real_opt_step_functor

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_REAL_OPT_STEP_FUNCTOR_HPP
