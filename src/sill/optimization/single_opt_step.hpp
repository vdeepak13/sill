
#ifndef SILL_SINGLE_OPT_STEP_HPP
#define SILL_SINGLE_OPT_STEP_HPP

#include <sill/optimization/real_opt_step.hpp>
#include <sill/optimization/void_step_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for single_opt_step.
   */
  struct single_opt_step_parameters {

    //! Types of choices of step size eta.
    enum eta_choice_enum {FIXED_ETA};

    /**
     * Choice of step size eta.
     *  (default = FIXED_ETA)
     */
    eta_choice_enum eta_choice;

    //! If using FIXED_ETA, use this value (> 0).
    //!  (default = .1)
    double fixed_eta;

    single_opt_step_parameters()
      : eta_choice(FIXED_ETA), fixed_eta(.1) { }

    virtual ~single_opt_step_parameters() { }

    bool valid() const {
      switch (eta_choice) {
      case FIXED_ETA:
        if (fixed_eta <= 0)
          return false;
        break;
      default:
        return false;
      }
      return true;
    }

  }; // struct single_opt_step_parameters

  /**
   * Chooses a step size eta for real-valued optimization not based on the
   * objective, gradient, etc.  This can be:
   *  - fixed eta
   *  - decreasing eta
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  class single_opt_step
    : public real_opt_step {

    single_opt_step_parameters params;

    // Public methods
    //==========================================================================
  public:

    single_opt_step(const single_opt_step_parameters& params)
      : params(params) {
      assert(params.valid());
    }

    /**
     * Execute one optimization step.
     */
    void step(const real_opt_step_functor& f) {
      // do nothing
    }

    //! Step size chosen.
    double eta() const {
      switch(params.eta_choice) {
      case single_opt_step_parameters::FIXED_ETA:
        return params.fixed_eta;
      default:
        assert(false);
        return 0;
      }
    }

    //! NOTE: This is never valid (for this type of real_opt_step)!
    double objective() const {
      assert(false); // This should not be called.
      return std::numeric_limits<double>::infinity();
    }

    //! Returns true if objective() returns a valid value.
    bool valid_objective() const {
      return false;
    }

    //! Number of calls to objective functor.
    size_t calls_to_objective() const {
      return 0;
    }

  };  // class single_opt_step

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SINGLE_OPT_STEP_HPP
