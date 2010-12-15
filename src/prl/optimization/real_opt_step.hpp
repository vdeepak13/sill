
#ifndef PRL_REAL_OPT_STEP_HPP
#define PRL_REAL_OPT_STEP_HPP

#include <prl/optimization/real_opt_step_functor.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Interface for algorithms which execute steps in real-valued optimization
   * algorithms (after the step direction has been chosen).
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  class real_opt_step {

    // Public methods
    //==========================================================================
  public:

    real_opt_step() { }

    virtual ~real_opt_step() { }

    /**
     * Execute one optimization step, setting eta (and possibly objective).
     *
     * @param f  Functor which computes the objective for a given step size eta
     *           (and possibly the gradient as well).
     */
    virtual void step(const real_opt_step_functor& f) = 0;

    //! Step size chosen.
    virtual double eta() const = 0;

    //! Objective value at chosen step size.
    //! NOTE: This may not be valid; call valid_objective() to check.
    virtual double objective() const = 0;

    //! Returns true if objective() returns a valid value.
    virtual bool valid_objective() const = 0;

    //! Number of calls to objective functor.
    virtual size_t calls_to_objective() const = 0;

  };  // class real_opt_step

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_REAL_OPT_STEP_HPP
