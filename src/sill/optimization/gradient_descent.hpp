
#ifndef SILL_GRADIENT_DESCENT_HPP
#define SILL_GRADIENT_DESCENT_HPP

#include <sill/optimization/gradient_method.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Parameters for gradient_descent class.
  struct gradient_descent_parameters
    : public gradient_method_parameters {

    typedef gradient_method_parameters base;

    gradient_descent_parameters(const gradient_method_parameters& gm_params =
                                gradient_method_parameters())
      : base(gm_params) { }

  }; // struct gradient_descent_parameters

  /**
   * Batch and stochastic gradient descent
   *  - for unconstrained nonlinear optimization
   *  - minimizes the given objective
   *
   * This tries to minimize the Objective, parametrized by an OptVector,
   * by calling a Gradient functor to compute a descent direction and then
   * taking a step in that direction.
   * Multiple algorithms may be implemented using this class:
   *  - Gradient: This may be an exact (batch) gradient computation,
   *              or it may be stochastic.
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
  class gradient_descent
    : public gradient_method<OptVector, Objective, Gradient> {

    // Protected data and methods
    //==========================================================================
  protected:

    typedef gradient_method<OptVector, Objective, Gradient> base;

    // Import from base class:
    using base::grad_functor;
    using base::x_;
    using base::direction_;
    using base::iteration_;

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor.
     * @param x_   Pre-allocated and initialized variables being optimized over.
     */
    gradient_descent(const Objective& obj_functor,
                    const Gradient& grad_functor,
                    OptVector& x_,
                    const gradient_descent_parameters& params
                    = gradient_descent_parameters())
      : base(obj_functor, grad_functor, x_, params) {
    }

    //! Perform one step.
    //! @return  False if at optimum.
    bool step() {
      // Compute the direction
      grad_functor.gradient(direction_, x_);
      direction_ *= -1;
      // Do a line search
      return base::run_line_search(direction_);
    }

  }; // class gradient_descent

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_GRADIENT_DESCENT_HPP
