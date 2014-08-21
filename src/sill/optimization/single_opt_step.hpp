#ifndef SILL_SINGLE_OPT_STEP_HPP
#define SILL_SINGLE_OPT_STEP_HPP

#include <sill/optimization/real_opt_step.hpp>
#include <sill/optimization/void_step_functor.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for single_opt_step.
   */
  struct single_opt_step_parameters {

    /**
     * Types of choices of step size eta (where t = iteration count).
     *  - FIXED_ETA: eta_t = init_eta
     *  - DECREASING_ETA: eta_t = init_eta * shrink_eta^t
     */
    enum eta_choice_enum {FIXED_ETA, DECREASING_ETA };

    /**
     * Choice of step size eta.
     *  (default = FIXED_ETA)
     */
    eta_choice_enum eta_choice;

    //! Initial step size value (> 0).
    //!  (default = .1)
    double init_eta;

    //! (For DECREASING_ETA)
    //! Discount factor in (0,1] by which eta is shrunk each round.
    //!  (default = .999)
    double shrink_eta;

    single_opt_step_parameters()
      : eta_choice(FIXED_ETA), init_eta(.1), shrink_eta(.999) { }

    virtual ~single_opt_step_parameters() { }

    //! Set shrink_eta based on init_eta and the given number of iterations.
    //! This sets shrink_eta so that eta will be about 0.0001 smaller
    //! at the end of optimization.
    void set_shrink_eta(size_t num_iterations);

    bool valid() const {
      if (eta_choice > DECREASING_ETA)
        return false;
      if (init_eta <= 0)
        return false;
      if (shrink_eta <= 0 || shrink_eta > 1)
        return false;
      return true;
    }

    void save(oarchive& ar) const;

    void load(iarchive& ar);

    void print(std::ostream& out, const std::string& line_prefix = "") const;

  }; // struct single_opt_step_parameters

  oarchive&
  operator<<(oarchive& a,
             single_opt_step_parameters::eta_choice_enum eta_choice);

  iarchive&
  operator>>(iarchive& a,
             single_opt_step_parameters::eta_choice_enum& eta_choice);

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
      : params(params), current_eta(0) {
      assert(params.valid());
      if (params.eta_choice == single_opt_step_parameters::DECREASING_ETA) {
        // step() will be called before eta(), so we need to adjust for that.
        current_eta = params.init_eta / params.shrink_eta;
      }
    }

    /**
     * Execute one optimization step.
     */
    void step(const real_opt_step_functor& f) {
      switch(params.eta_choice) {
      case single_opt_step_parameters::FIXED_ETA:
        // do nothing
        break;
      case single_opt_step_parameters::DECREASING_ETA:
        current_eta *= params.shrink_eta;
        // TO DO: Make sure the step size does not get small enough to have
        //        numerical issues.
        break;
      default:
        assert(false);
      }
    }

    //! Step size chosen.
    double eta() const {
      switch(params.eta_choice) {
      case single_opt_step_parameters::FIXED_ETA:
        return params.init_eta;
      case single_opt_step_parameters::DECREASING_ETA:
        return current_eta;
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

    // Private data
    //==========================================================================

    //! (used for DECREASING_ETA)
    double current_eta;

  };  // class single_opt_step

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SINGLE_OPT_STEP_HPP
