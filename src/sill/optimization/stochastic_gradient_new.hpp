
#ifndef SILL_STOCHASTIC_GRADIENT_NEW_HPP
#define SILL_STOCHASTIC_GRADIENT_NEW_HPP

#include <sill/optimization/gradient_method.hpp>
#include <sill/optimization/void_objective.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for stochastic_gradient_new.
   *
   * Note that some options from gradient_method_parameters cannot be changed
   * (and are ignored if set incorrectly):
   *  - step_type (fixed to SINGLE_OPT_STEP)
   *  - ls_params (not used)
   *  - ls_stopping (not used)
   *
   * By default, this adjusts single_opt_step_params to use DECREASING_ETA
   * with init_eta = 1 and shrink_eta =~ .999 (which is reasonable for about
   * 10000 iterations).
   */
  struct stochastic_gradient_new_parameters
    : public gradient_method_parameters {

    typedef gradient_method_parameters base;

    stochastic_gradient_new_parameters();

    explicit
    stochastic_gradient_new_parameters
    (const gradient_method_parameters& gm_params);

    //! This method is called by stochastic_gradient_new to correct options in
    //! gradient_method_parameters.
    void set_defaults();

    bool valid() const;

    void print(std::ostream& out) const;

  }; // struct stochastic_gradient_new_parameters

  std::ostream&
  operator<<(std::ostream& out,
             const stochastic_gradient_new_parameters& params);

  /**
   * Stochastic gradient descent
   *  - for unconstrained nonlinear optimization
   *  - MINIMIZES the objective
   *
   * This class is similar to gradient_descent, but it restricts some parameter
   * choices.  Specifically, it does not use line searches.
   *
   * @tparam OptVector   Datatype which stores the optimization variables.
   * @tparam Gradient    Type of functor which computes the gradient.
   *                      (This will presumably be a stochastic estimate.)
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  template <typename OptVector, typename Gradient>
  class stochastic_gradient_new
    : public gradient_method<OptVector, void_objective<OptVector>, Gradient> {

    concept_assert((sill::OptimizationVector<OptVector>));
    concept_assert((sill::GradientFunctor<Gradient, OptVector>));

    // Public types
    //==========================================================================
  public:

    typedef gradient_method<OptVector,void_objective<OptVector>,Gradient> base;

    // Public methods
    //==========================================================================

    /**
     * Constructor for stochastic_gradient_new.
     * @param x_   Pre-allocated and initialized optimization variables.
     */
    stochastic_gradient_new
    (const Gradient& grad_functor, OptVector& x_,
     const stochastic_gradient_new_parameters& params
     = stochastic_gradient_new_parameters())
      : base(void_obj_functor, grad_functor, x_, params) {
      assert(params.valid());
    }

    //! Perform one step.
    //! @return  Always true.
    bool step() {
      // Note: This method does not use gradient_method::run_line_search,
      //       which incurs extra overhead.
      grad_functor.gradient(direction_, x_);

      if (params.debug > 0) {
        double step_magnitude(direction_.L2norm());
        if (!rls_valid_step_magnitude_(step_magnitude))
          return false;
      }

      assert(step_ptr);
      step_ptr->step(*void_step_functor_ptr);
      x_ -= direction_ * step_ptr->eta();

      ++iteration_;

      return true;
    }

    // Protected data and methods
    //==========================================================================
  protected:

    // Import from base class:
    using base::grad_functor;
    using base::x_;
    using base::direction_;
    using base::iteration_;
    using base::step_ptr;
    using base::params;
    using base::void_step_functor_ptr;

    using base::rls_valid_step_magnitude_;

    void_objective<OptVector> void_obj_functor;

  }; // class stochastic_gradient_new

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_STOCHASTIC_GRADIENT_NEW_HPP
