#ifndef SILL_BRACKETING_LINE_SEARCH_HPP
#define SILL_BRACKETING_LINE_SEARCH_HPP

#include <sill/math/constants.hpp>
#include <sill/optimization/opt_step.hpp>
#include <sill/serialization/serialize.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for bracketing_line_search.
   * \ingroup optimization_algorithms
   */
  template <typename RealType>
  struct bracketing_line_search_parameters {

    /**
     * If the change in the objective is less than this value (>=0),
     * then the line search will declare convergence.
     */
    RealType convergence;

    /**
     * Value (>1) by which the step size is multiplied / divided by
     * when searching for the initial brackets.
     */
    RealType multiplier;

    /**
     * If the step size reaches this value (>=0), the line search will throw 
     * an opt_step_not_found exception.
     */
    RealType min_step;

    /**
     * If the step size reaches this value (>=0), the line search will throw
     * an opt_step_not_found exception.
     */
    RealType max_step;

    bracketing_line_search_parameters(RealType convergence = 1e-6,
                                      RealType multiplier = 2.0,
                                      RealType min_step = 1e-10,
                                      RealType max_step = 1e+10)
      : convergence(convergence),
        multiplier(multiplier),
        min_step(min_step),
        max_step(max_step) {
      assert(valid());
    }

    bool valid() const {
      return convergence >= 0.0 && multiplier > 1.0 && min_step >= 0.0;
    }

    void save(oarchive& ar) const {
      ar << convergence << multiplier << min_step;
    }

    void load(iarchive& ar) {
      ar >> convergence >> multiplier >> min_step;
    }

    friend std::ostream&
    operator<<(std::ostream& out, const bracketing_line_search_parameters& p) {
      out << p.convergence << ' ' << p.multiplier << ' ' << p.min_step;
      return out;
    }

  }; // struct bracketing_line_search_parameters

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
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class bracketing_line_search : public real_opt_step {

    // Public types
    //==========================================================================
  public:
    typedef typename Vec::value_type real_type;
    typedef bracketing_line_search_parameters<real_type> param_type;
    typedef wolfe_conditions_parameters<real_type> wolfe_param_type;
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    // Public functions
    //==========================================================================
  public:
    /**
     * Constructs an object that performs line search with objective
     * function alone. This is preferable when evaluating the objective
     * is much cheaper than the gradient.
     */
    explicit bracketing_line_search(const objective_fn& objective,
                                    const param_type& params = param_type())
      : f_(objective),
        params_(params) {
      assert(params.valid());
    }
    
    /**
     * Constructs an object that performs line search using gradient.
     * This is preferable when evaluating the gradient is cheap.
     */
    bracketing_line_search(const objective_fn& objective,
                           const gradient_fn& gradient,
                           const param_type& params = param_type())
      : f_(objective, gradient),
        params_(params) {
      assert(params.valid());
    }

    /**
     * Constructs an objective tht performs line search using gradient
    
    real_type apply(Vec& x, const Vec& direction) {
      f_.line(&x, &direction);
      real_type t = gradient_ ? find_step_derivative() : find_step_objective();
      real_type value = f_(t);
      x = f_.last_input();
      return value;
    }

    // Private data and functions
    //==========================================================================
  private:
    /**
     * Returns the step size that makes derivative approximately zero.
     */
    real_type find_step_derivative() {
      // make sure that the left derivative is < 0
      real_type left = 0.0;
      real_type left_deriv = f_.derivative(left);
      if (left_deriv > 0.0) {
        throw std::invalid_argument("The function is increasing along the "
                                    "specified direction");
      } else if (left_deriv == 0.0) {
        return left;
      }

      // find the right bound s.t. the right derivative >= 0
      real_type right = 1.0;
      real_type right_deriv = f_.derivative(right);
      while (right_deriv < 0.0) {
        right *= params_.multiplier;
        if (right > params_.max_step) {
          throw std::opt_step_not_found("Could not find right bound <= " +
                                        to_string(params_.max_step));
        }
        right_deriv = f_.derivative(right);
      }

      // do binary search until we shrink the bracket sufficiently
      // convergence must be < min_step, so that we don't
      // converge too early before we actually moved the left pointer
      // alternative: force left to increase before convergence
      while (right - left > params_.convergence) {
        real_type mid = (left + right) / 2.0;
        real_type mid_deriv = f_.derivative(mid);
        if (mid_deriv < 0.0) {
          left = mid;
          left_deriv = mid_deriv;
        } else {
          right = mid;
          right_deriv = mid_deriv;
        }
        if (right < params_.min_step) {
          throw std::opt_step_not_found("Step size is too small");
        }
      }

      // the left side of the bracket is guaranteed to have lower objective
      // than the start
      return left;
    }

    /**
     * Returns the step size that approximately minimizes the objective value.
     * This is achieved by maintaining three step sizes, left, mid, and right,
     * such that f(mid) < f(0), f(mid) < f(left), f(mid) < f(right).
     * We declare convergence once |right-left| becomes sufficiently small,
     * returning mid as the result. This function works for both convex
     * and non-convex, multi-modal objectives and is guaranteed to decrease
     * the objective value.
     */
    real_type find_step_objective() {
      typedef typename line_function<Vec>::value_type value_type;
      value_type left  = f_.value(0.0);
      value_type mid   = f_.value(1.0);
      value_type right = mid;
      
      // identify the initial bracket
      if (right.obj > left.obj) {
        mid = f_.value(1.0 / params_.multiplier);
        while (mid.obj > left.obj) {
          right = mid;
          mid = f_.value(mid.pos / params_.multiplier);
          if (right.pos < params_.min_step) {
            throw opt_step_not_found("Step size too small in bounding");
          }
        }
      } else {
        right = f_.value(params_.multiplier);
        while (right.obj < mid.obj) {
          left = mid;
          mid = right;
          right = f_.value(right.pos * params_.multiplier);
          if (right.pos > params_.max_step) {
            throw opt_step_not_found("Step size too large in bounding");
          }
        }
      }

      // do binary search while maintaining the invariant
      while (right.pos - left.pos > params_.convergence) {
        value_type mid_left = f_.value((left.pos + mid.pos) / 2.0);
        value_type mid_right = f_.value((mid.pos + right.pos) / 2.0);
        if (mid_left.obj > mid.obj && mid_right.obj > mid.obj) {
          left = mid_left;
          right = mid_right;
        } else if (mid_left.obj < mid_right.obj) {
          right = mid;
          mid = mid_left;
        } else {
          left = mid;
          mid = mid_right;
        }
        if (right.pos < params_.min_step) {
          throw std::opt_step_not_found("Step size is too small");
        }
      }

      return left;
    }

    line_function<Vec> f_;
    objective_fn objective_;
    gradient_fn gradient_;
    param_type params_;
    size_t bounding_steps_;
    size_t searching_steps_;






    line_search_parameters params;

    //! Serves as the middle eta during the search and the final eta at the end.
    double mid_eta;

    //! Corresponds to obj_functor(mid_eta).
    double mid_obj;

    size_t bounding_steps_;

    size_t searching_steps_;

    size_t calls_to_objective_;

    //! Returns true if val1 is within convergence_zero of val2.
    bool approx_eq(double val1, double val2) const {
      return (fabs(val1 - val2) <= params.convergence_zero);
    }

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
           (front_obj) (mid_obj) (back_obj)
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
      if (front_obj == inf()) {
        if (params.debug > 0)
          std::cerr << "WARNING: line_search failed since initial objective = "
                    << front_obj << " (for eta = " << front_eta << ")"
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

      // Do a binary search within the bounds.
      while (!approx_eq(back_obj, front_obj) ||
             !approx_eq(back_obj, mid_obj) ||
             !approx_eq(mid_obj, front_obj)) {
        if (approx_eq(back_eta, mid_eta) || approx_eq(front_eta, mid_eta)) {
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

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_OPTIMIZATION_LINE_SEARCH_HPP
