// I STOPPED WRITING THIS PARTWAY.  FINISH IT SOMETIME IF IT MAKES SENSE.

#ifndef SILL_FISTA_HPP
#define SILL_FISTA_HPP

#include <sill/optimization/gradient_method.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Parameters for fista class.
  struct fista_parameters
    : public gradient_method_parameters {

    typedef gradient_method_parameters base;

    fista_parameters(const gradient_method_parameters& gm_params =
                     gradient_method_parameters())
      : base(gm_params) { }

  }; // struct fista_parameters

  /**
   * Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
   *  (Beck and Teboulle, 2009)
   *
   *
   * For FISTA with constant stepsize,
   *  - Let L = Lipschitz constant of gradient of objective.
   *  - Set parameters:
   *     - step_type = parameters::SINGLE_OPT_STEP
   *     - single_opt_step_params.eta_choice = parameters::FIXED_ETA
   *     - single_opt_step_params.init_eta = L
   * For FISTA with backtracking,
   *  - TO DO
   *
   *
   * ** TO DO: EDIT DOCUMENTATION BELOW! (COPIED FROM GRADIENT_DESCENT) ***
   *
   *  - for unconstrained nonlinear optimization
   *  - MINIMIZES the objective
   *
   * This tries to minimize the Objective, parametrized by an OptVector,
   * by calling a Gradient functor to compute a descent direction and then
   * taking a step in that direction.
   * Multiple algorithms may be implemented using this class:
   *  - Gradient: This may be an exact (batch) gradient computation,
   *              or it may be stochastic.  For stochastic, consider using
   *              the stochastic_gradient class, which helpfully restricts
   *              certain options.
   *  - step: Use the parameter settings to choose how progress is
   *          made in descent directions.  A step may be done via
   *          single steps or via line searches.
   *
   * @tparam OptVector   Datatype which stores the optimization variables.
   * @tparam Objective   Type of functor which computes the objective value.
   * @tparam Gradient    Type of functor which computes the gradient.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  template <typename OptVector, typename Objective, typename Gradient>
  class fista
    : public gradient_method<OptVector, Objective, Gradient> {

    // Public types
    //==========================================================================
  public:

    typedef gradient_method<OptVector, Objective, Gradient> base;

    // Public methods
    //==========================================================================

    /**
     * Constructor.
     * @param x_   Pre-allocated and initialized optimization variables.
     */
    fista(const Objective& obj_functor,
          const Gradient& grad_functor,
          OptVector& x_,
          const fista_parameters& params = fista_parameters())
      : base(obj_functor, grad_functor, x_, params) {
    }

    //! Perform one step.
    //! @return  False if at optimum.
    bool step() {
      // Compute the direction
      grad_functor.gradient(direction_, x_);
      direction_ *= -1;
      // Do a line search
      return base::run_line_search();
    }

    // Protected data and methods
    //==========================================================================
  protected:

    // Import from base class:
    using base::grad_functor;
    using base::x_;
    using base::direction_;
    using base::iteration_;

  }; // class fista

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_FISTA_HPP
