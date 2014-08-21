#ifndef SILL_LINE_SEARCH_WITH_GRAD_HPP
#define SILL_LINE_SEARCH_WITH_GRAD_HPP

#include <sill/optimization/line_search.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for doing a line search from a point x in a given direction
   * to minimize an objective w.r.t. a non-negative scale parameter eta.
   * This extends line_search so that searches take advantage of gradient
   * computations; it is useful when it is not much more expensive to compute
   * the gradient when computing the objective.
   *
   * This can handle:
   *  - Convex objectives (using bracketed binary search)
   *  - Non-convex objectives (using decreasing binary search from 1)
   *
   * Line searches check the objective functor to see if they can stop early.
   * This option permits approximate line search using the Wolfe conditions
   * or similar criteria.
   *
   * @todo Modify this so that the caller can call a step() function.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   *
   *  - 2) Line search using objective and gradient:
   *     - Same as line search only using the objective, except that this can
   *       use about 1/2 as many calls to the objective/gradient.
   *     - This is best when it is cheap to compute the gradient.
   */
  class line_search_with_grad
    : public line_search {

    typedef line_search base;

    //! Max eta allowed.
    //! Set to std::numeric_limits<double>::max() / (2. * ls_eta_mult).
    double ls_MAXIMUM_ETA;

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor which does not run a line search but sets the parameters.
     */
    explicit
    line_search_with_grad
    (const line_search_parameters& params = line_search_parameters())
      : base(params),
        ls_MAXIMUM_ETA
        (std::numeric_limits<double>::max() / (2. * params.ls_eta_mult)) {
    }

    /**
     * Run line search using the given objective-gradient functor.
     * This method is best when it is cheap to compute the gradient along with
     * the objective.
     *
     * See the base class' step() documentation for more info.
     */
    void step(const real_opt_step_functor& step_functor) {
      if (!step_functor.valid_gradient()) {
        throw std::runtime_error("line_search_with_grad::step() was called, but the step functor does not implement the gradient() method.");
      }

      /* This uses bracketing variables and corresponding objective values:
          [ front_eta < middle_eta < mid_eta ]
           (front_obj) (middle_obj)  (mid_obj)
          (front_grad) (middle_grad) (mid_grad)
       */

      init_search();
      mid_eta = params.ls_init_eta;
      if (ls_helper(mid_obj, mid_eta, step_functor))
        return;
      double mid_grad(step_functor.gradient(mid_eta));

      // Find bounds [front_eta, mid_eta_] s.t.
      //  gradient(front_eta) > 0 and gradient(mid_eta_) < 0,
      //  but watch the objective values too (for non-convex problems).
      double front_eta(0.);
      double front_obj;
      if (ls_helper(front_obj, front_eta, step_functor)) {
        if (params.debug > 0)
          std::cerr << "line_search_with_grad converged at eta = 0."
                    << "  Why did you call line_search_with_grad?"
                    << std::endl;
        return;
      }
      double front_grad(step_functor.gradient(front_eta));
      if (front_grad >= 0) {
        mid_eta = front_eta;
        mid_obj = front_obj;
        return;
      }
      while (mid_grad < 0) {
        if (mid_grad < front_grad) {
          if (params.debug > 0)
            std::cerr << "line_search_with_grad::step(obj,grad) found that"
                      << " the objective is not convex." << std::endl;
        }
        if (mid_obj < front_obj) {
          front_eta = mid_eta;
          front_obj = mid_obj;
          front_grad = mid_grad;
          mid_eta *= params.ls_eta_mult;
          if (mid_eta > ls_MAXIMUM_ETA) {
            throw std::runtime_error
              ("line_search_with_grad::step() reached eta > ls_MAXIMUM_ETA.");
          }
          if (ls_helper(mid_obj, mid_eta, step_functor))
            return;
          mid_grad = step_functor.gradient(mid_eta);
        } else {
          mid_eta /= params.ls_eta_mult;
          if (ls_helper(mid_obj, mid_eta, step_functor))
            return;
          mid_grad = step_functor.gradient(mid_eta);
	  if (mid_eta
              < params.convergence_zero * params.ls_eta_zero_multiplier) {
	    if (params.debug > 0)
	      std::cerr << "line_search_with_grad::step(obj,grad) exited"
			<< " during bounding search with eta <"
			<< " convergence_zero * ls_eta_zero_multiplier = "
			<< (params.convergence_zero
                            * params.ls_eta_zero_multiplier)
			<< std::endl;
	    return;
	  }
        }
        ++bounding_steps_;
      }

      // Do a binary search to narrow the bounds.
      double middle_eta = (mid_eta + front_eta) / 2.;
      double middle_obj;
      if (ls_helper(middle_obj, middle_eta, step_functor))
        return;
      double middle_grad = step_functor.gradient(middle_eta);
      if ((middle_grad < front_grad) || (mid_grad < middle_grad)) {
        if (params.debug > 0)
          std::cerr << "line_search_with_grad::step(obj,grad) found that"
                    << " the objective is not convex." << std::endl;
      }
      do {
        // Check objective first (to handle non-convex problems).
        if ((middle_obj > front_obj) || (middle_obj > mid_obj)) {
          if (front_obj <= mid_obj) {
            mid_eta = middle_eta;
            mid_obj = middle_obj;
            mid_grad = middle_grad;
          } else {
            front_eta = middle_eta;
            front_obj = middle_obj;
            front_grad = middle_grad;
          }
        } else {
          // Since middle_obj < front_obj,mid_obj, check gradients.
          if (middle_grad > 0) {
            mid_eta = middle_eta;
            mid_obj = middle_obj;
            mid_grad = middle_grad;
          } else if (middle_grad < 0) {
            front_eta = middle_eta;
            front_obj = middle_obj;
            front_grad = middle_grad;
          } else {
            mid_eta = middle_eta;
            mid_obj = middle_obj;
            return;
          }
        }
        /*
        double scaled_step_size(mid_eta * params.ls_step_magnitude);
        if (scaled_step_size <
	    params.convergence_zero * params.zero_eta_multiplier)
          break;
        */
        middle_eta = (mid_eta + front_eta) / 2.;
        if (ls_helper(middle_obj, middle_eta, step_functor))
          return;
        middle_grad = step_functor.gradient(middle_eta);
        if ((middle_grad < front_grad) || (mid_grad < middle_grad)) {
          if (params.debug > 0)
            std::cerr << "line_search_with_grad::step(obj,grad) found that"
                      << " the objective is not convex." << std::endl;
        }
        ++searching_steps_;
        if (params.debug > 0)
          std::cerr << "   [" << front_eta << ", " << middle_eta << ", "
                    << mid_eta << "]-->["
                    << front_obj << ", " << middle_obj << ", " << mid_obj
                    << "]" << std::endl;
      } while ((fabs(mid_obj - front_obj) > params.convergence_zero) ||
               (fabs(mid_obj - middle_obj) > params.convergence_zero) ||
               (fabs(middle_obj - front_obj) > params.convergence_zero));
      if (params.debug > 0)
        std::cerr << "  Chose: eta = " << mid_eta << ", objective = " << mid_obj
                  << std::endl;
    } // step()

  }; // class line_search_with_grad

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LINE_SEARCH_WITH_GRAD_HPP
