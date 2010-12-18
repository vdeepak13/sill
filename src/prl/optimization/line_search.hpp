
#ifndef SILL_OPTIMIZATION_LINE_SEARCH_HPP
#define SILL_OPTIMIZATION_LINE_SEARCH_HPP

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <boost/type_traits/is_same.hpp>

#include <sill/optimization/real_opt_step.hpp>
#include <sill/optimization/real_opt_step_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Parameters for line_search.
  struct line_search_parameters {

    //! If the change in objective is less than this value,
    //! then the line search will declare convergence.
    //!  (default = .000001)
    double convergence_zero;

    //! If the step size eta (times the ls_step_magnitude option) is less than
    //! ls_eta_zero_multiplier * convergence_zero,
    //! then the line search will declare convergence.
    //!  (default = .0000001)
    double ls_eta_zero_multiplier;

    //! The magnitude of the step size when eta == 1; this allows the search
    //! to determine convergence in terms of the absolute value of the
    //! actual step size, rather than in terms of the multiplier eta.
    //!  (default = 1)
    double ls_step_magnitude;

    //! Initial step size multiplier to try.
    //!  (default = 1)
    double ls_init_eta;

    //! Value (> 1) by which the step size multiplier eta is
    //! multiplied/divided by on each step of the search.
    //!  (default = 2)
    double ls_eta_mult;

    /**
     * Print debugging info:
     *  - 0: none (default)
     *  - 1: some
     *  - higher values: revert to highest level of debugging
     */
    size_t debug;

    line_search_parameters()
      : convergence_zero(.000001), ls_eta_zero_multiplier(.0000001),
        ls_step_magnitude(1), ls_init_eta(1), ls_eta_mult(2),
        debug(0) {
    }

    virtual ~line_search_parameters() { }

    bool valid() const {
      if (convergence_zero < 0)
        return false;
      if (ls_eta_zero_multiplier < 0)
        return false;
      if (ls_step_magnitude <= 0)
        return false;
      if (ls_init_eta <= 0)
        return false;
      if (ls_eta_mult <= 1)
        return false;
      return true;
    }

    void print(std::ostream& out) const {
      out << "convergence_zero=" << convergence_zero << "\n"
          << "ls_eta_zero_multiplier=" << ls_eta_zero_multiplier << "\n"
          << "ls_step_magnitude=" << ls_step_magnitude << "\n"
          << "ls_init_eta=" << ls_init_eta << "\n"
          << "ls_eta_mult=" << ls_eta_mult << "\n";
    }

  }; // struct line_search_parameters

  /**
   * Class for doing a line search from a point x in a given direction
   * to minimize an objective w.r.t. a non-negative scale parameter eta.
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
   */
  class line_search
    : public real_opt_step {

    // Protected data and methods
    //==========================================================================
  protected:

    line_search_parameters params;

    //! Serves as the middle eta during the search and the final eta at the end.
    double mid_eta;

    //! Corresponds to obj_functor(mid_eta).
    double mid_obj;

    size_t bounding_steps_;

    size_t searching_steps_;

    size_t calls_to_objective_;

    //! Helper used by run_line_search().
    //! Sets obj to objective(eta); returns true if line search can stop early.
    bool ls_helper
    (double& obj, double eta, const real_opt_step_functor& obj_functor) {
      ++calls_to_objective_;
      obj = obj_functor.objective(eta);
      if (obj_functor.stop_early()) {
        mid_eta = eta;
        mid_obj = obj;
        return true;
      }
      return false;
    }

    void init_search() {
      if (!params.valid()) {
        std::cerr << "line_search_parameters:\n";
        params.print(std::cerr);
        std::cerr << std::endl;
        throw std::invalid_argument
          ("line_search::run_line_search() called on line_search with invalid parameters.");
      }
      if (params.debug > 0)
        std::cerr << "line_search() with ls_init_eta = " << params.ls_init_eta
                  << ", ls_eta_mult = " << params.ls_eta_mult << std::endl;
      // Reset stored values for new search.
      bounding_steps_ = 0;
      searching_steps_ = 0;
      calls_to_objective_ = 0;
    }

    /*
    //! Choose the next step length to try during the bounding phase.
    double choose_eta_bounding() const {
      return -1;
    }

    //! Choose the next step length to try during the phase which shrinks
    //! the bounds to a particular eta.
    */

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor which does not run a line search but sets the parameters.
     */
    explicit
    line_search(const line_search_parameters& params = line_search_parameters())
      : params(params) {
    }

    /**
     * Run line search using only the objective.
     * This method is best when the objective is much cheaper to compute than
     * the gradient.
     *
     * This uses the parameters stored in this class.
     * For CONVEX problems, this uses bracketed binary search:
     *  - Start with ls_init_eta.
     *  - Find bounds within which the optimal eta must lie.
     *  - Do binary search within those bounds.
     *     - If the objectives at [eta_min, eta_mid, eta_max] are ever equal,
     *       stop the search (since all intermediate points must be equal for
     *       convex objectives).
     * For NON_CONVEX problems, this uses decreasing binary search from 1:
     *  - Start with ls_init_eta (which should be left as the default 1).
     *  - Decrease eta by 1/2 each iteration until stopping conditions are met.
     * 
     * Note: This is virtual so that line_search_with_grad (a child class)
     *       can override this method with one which takes advantage of the
     *       gradient.
     */
    virtual void step(const real_opt_step_functor& obj_functor) {
      init_search();
      mid_eta = params.ls_init_eta;
      if (ls_helper(mid_obj, mid_eta, obj_functor))
        return;

      /* This uses bracketing variables and corresponding objective values:
          [ front_eta < mid_eta < back_eta ]
           (front_obj) (middle_obj)  (mid_obj)
       */

      // Find bounds [front_eta, back_eta]
      double front_eta(0.);
      double front_obj;
      if (ls_helper(front_obj, front_eta, obj_functor)) {
        if (params.debug > 0)
          std::cerr << "line_search converged at eta = 0."
                    << "  Why did you call line_search?"
                    << std::endl;
        return;
      }
      double back_obj;
      double back_eta;
      if (front_obj < mid_obj) {
        // Then we can use bounds [front_obj, mid_obj]
        back_eta = mid_eta;
        back_obj = mid_obj;
      } else {
        if (ls_helper(back_obj, params.ls_eta_mult * mid_eta, obj_functor))
          return;
        while(mid_obj > back_obj) {
          front_eta = mid_eta;
          front_obj = mid_obj;
          mid_eta *= params.ls_eta_mult;
          mid_obj = back_obj;
          if (ls_helper(back_obj, params.ls_eta_mult * mid_eta, obj_functor))
            return;
          ++bounding_steps_;
        }
        back_eta = params.ls_eta_mult * mid_eta;
      }
      if (params.debug > 0)
        std::cerr << "  Using bounds [front_eta, back_eta] = ["
                  << front_eta << ", " << back_eta << "]\n"
                  << "  Now searching via [front_eta, mid_eta, back_eta]-->"
                  << "[front_obj, mid_obj, back_obj]" << std::endl;
      // To do: These 2 lines could be merged with the above for efficiency.
      mid_eta = (back_eta + front_eta) / 2.;
      if (ls_helper(mid_obj, mid_eta, obj_functor))
        return;

      // Do a binary search
      while ((fabs(back_obj - front_obj) > params.convergence_zero) ||
             (fabs(back_obj - mid_obj) > params.convergence_zero) ||
             (fabs(mid_obj - front_obj) > params.convergence_zero)) {
        if ((back_eta == mid_eta) || (front_eta == mid_eta)) {
          if (params.debug > 0)
            std::cerr << "line_search: step sizes eta converged, but the"
                      << " objective did not yet converge.  You may have"
                      << " numerical issues."
                      << std::endl;
          break;
        }
        /*
        double scaled_step_size(back_eta * params.ls_step_magnitude);
        if (scaled_step_size <
	    params.convergence_zero * params.ls_eta_zero_multiplier)
          break;
        */
        ++searching_steps_;
        if (params.debug > 0)
          std::cerr << "   [" << front_eta << ", " << mid_eta << ", "
                    << back_eta << "]-->["
                    << front_obj << ", " << mid_obj << ", " << back_obj
                    << "]" << std::endl;
        if (mid_obj > front_obj) {
          back_eta = mid_eta;
          back_obj = mid_obj;
          mid_eta = (mid_eta + front_eta)/2.;
          if (ls_helper(mid_obj, mid_eta, obj_functor))
            return;
          continue;
        }
        double leftmid_obj_;
        if (ls_helper(leftmid_obj_, (mid_eta + front_eta)/2., obj_functor))
          return;
        if (leftmid_obj_ < mid_obj) {
          back_eta = mid_eta;
          back_obj = mid_obj;
          mid_eta = (mid_eta + front_eta)/2.;
          mid_obj = leftmid_obj_;
          continue;
        }
        double rightmid_obj_;
        if (ls_helper(rightmid_obj_, (back_eta + mid_eta)/2., obj_functor))
          return;
        if (rightmid_obj_ < mid_obj) {
          front_eta = mid_eta;
          front_obj = mid_obj;
          mid_eta = (back_eta + mid_eta)/2.;
          mid_obj = rightmid_obj_;
          continue;
        }
        front_eta = (mid_eta + front_eta) / 2.;
        front_obj = leftmid_obj_;
        back_eta = (back_eta + mid_eta) / 2.;
        back_obj = rightmid_obj_;
      }
      if (params.debug > 0)
        std::cerr << "  Chose: eta = " << mid_eta << ", objective = " << mid_obj
                  << std::endl;
    } // step()

    //! Step size chosen.
    double eta() const {
      return mid_eta;
    }

    //! Objective value at chosen step size.
    double objective() const {
      return mid_obj;
    }

    //! Returns true if objective() returns a valid value.
    bool valid_objective() const {
      return true;
    }

    //! Number of steps taken in initial search for bounds on eta.
    size_t bounding_steps() const {
      return bounding_steps_;
    }

    //! Number of steps taken in binary search for eta.
    size_t searching_steps() const {
      return searching_steps_;
    }

    //! Number of calls to objective functor.
    size_t calls_to_objective() const {
      return calls_to_objective_;
    }

    //! Returns the parameters.
    const line_search_parameters& get_params() const {
      return params;
    }

    //! Returns the parameters.
    line_search_parameters& get_params() {
      return params;
    }

  }; // class line_search

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_OPTIMIZATION_LINE_SEARCH_HPP
