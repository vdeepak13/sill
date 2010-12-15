
#ifndef PRL_STOCHASTIC_GRADIENT_HPP
#define PRL_STOCHASTIC_GRADIENT_HPP

#include <boost/type_traits/is_same.hpp>

#include <prl/base/stl_util.hpp>
#include <prl/math/is_finite.hpp>
#include <prl/math/vector.hpp>
#include <prl/optimization/concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  //! Parameters for stochastic_gradient subclasses.
  struct stochastic_gradient_parameters {

    //! Value (>= 0) small enough to be considered 0 for numerical purposes.
    //!  (default = .00001)
    double convergence_zero;

    /**
     * Method for choosing step sizes.
     *  - 0: Use step size: init_step_size * step_multiplier ^ iteration
     *     (default)
     */
    size_t step_size_method;

    //! Initial step size (> 0).
    //!  (default = 1)
    double init_step_size;

    //! Step size multiplier (in (0, 1)).
    //!  (default = .99)
    double step_multiplier;

    /**
     * Debug mode:
     *  - 0: no debugging (default)
     *  - 1: some
     *  - 2: more
     *  - higher: revert to highest debugging mode
     */
    size_t debug;

    stochastic_gradient_parameters()
      : convergence_zero(.00001), step_size_method(0), init_step_size(1),
        step_multiplier(.99), debug(0) { }

    bool valid() const {
      if (convergence_zero < 0)
        return false;
      if (step_size_method > 0)
        return false;
      if (init_step_size <= 0)
        return false;
      if ((step_multiplier <= 0) || (step_multiplier >= 1))
        return false;
      return true;
    }

  }; // struct stochastic_gradient_parameters

  /**
   * Interface for stochastic gradient-based algorithms which choose directions
   * and then take step sizes (of decreasing size) in those directions.
   * This is for unconstrained nonlinear optimization,
   * and it tries to minimize the objective.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   * @tparam Gradient       Type of functor which computes the gradient.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  template <typename OptVector, typename Gradient>
  class stochastic_gradient {

    concept_assert((prl::OptimizationVector<OptVector>));
    concept_assert((prl::GradientFunctor<Gradient, OptVector>));

    // Protected data and methods
    //==========================================================================
  protected:

    //! Gradient functor
    const Gradient& grad_functor;

    //! Current values of variables being optimized over.
    OptVector& x_;

    //! From parameters:
    double convergence_zero;

    //! From parameters:
    size_t step_size_method;

    //! From parameters:
    double step_multiplier;

    //! From parameters:
    size_t debug;

    //! Update direction.
    //! When step() is called, this direction can be modified.
    OptVector direction_;

    //! Iteration number.
    size_t iteration_;

    //! Current step size.
    double current_step_size_;

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor for stochastic_gradient.
     * @param x_   Pre-allocated and initialized variables being optimized over.
     */
    stochastic_gradient
    (const Gradient& grad_functor, OptVector& x_,
     const stochastic_gradient_parameters& params)
      : grad_functor(grad_functor), x_(x_),
        convergence_zero(params.convergence_zero),
        step_size_method(params.step_size_method),
        step_multiplier(params.step_multiplier), debug(params.debug),
        direction_(x_.size(), 0),
        iteration_(0), current_step_size_(params.init_step_size) {
      assert(params.valid());
    }

    //! Perform one step.
    //! @return  Always true (but perhaps false eventually once we can
    //!          measure convergence).
    bool step() {
      grad_functor.gradient(direction_, x_);
      direction_ *= current_step_size_;
      x_ += direction_;
      current_step_size_ *= step_multiplier;
      // TO DO: Make sure the step size does not get small enough to have
      //        numerical issues.
      ++iteration_;
      return true;
    } // step()

    //! Current values of variables being optimized over.
    const OptVector& x() const {
      return x_;
    }

    //! Current iteration (from 0), i.e., number of iterations completed.
    size_t iteration() const {
      return iteration_;
    }

  }; // class stochastic_gradient

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_STOCHASTIC_GRADIENT_HPP
