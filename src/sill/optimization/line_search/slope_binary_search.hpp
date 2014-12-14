#ifndef SILL_SLOPE_BINARY_SEARCH_HPP
#define SILL_SLOPE_BINARY_SEARCH_HPP

#include <sill/global.hpp>
#include <sill/optimization/line_search/bracketing_line_search.hpp>
#include <sill/optimization/line_search/line_search.hpp>
#include <sill/optimization/line_search/line_search_failed.hpp>
#include <sill/optimization/line_search/wolfe_conditions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class that performs line search by finding the place where the slope
   * is approximately 0. For each invocation of step(), we maintain left
   * and right step size, such that g(left) < 0 and g(right) > 0. A local
   * minimum of the function must then necessarily be located between left
   * and right. The convergence is declared either when |right-left| becomes
   * sufficiently small, or whern the optional Wolfe conditions are satisfied.
   *
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class slope_binary_search : public line_search<Vec> {

    // Public types
    //==========================================================================
  public:
    typedef typename Vec::value_type real_type;
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;
    typedef bracketing_line_search_parameters<real_type> param_type;
    typedef typename wolfe_conditions<real_type>::param_type wolfe_param_type;

    // Public functions
    //==========================================================================
  public:
    /**
     * Constructs the line search object for the given objective and gradient.
     * The search stops when the bracket becomes sufficiently small.
     */
    slope_binary_search(const objective_fn& objective,
                        const gradient_fn& gradient,
                        const param_type& params = param_type())
      : f_(objective, gradient),
        params_(params) {
      assert(params.valid());
    }

    /**
     * Constructs the line saerch object for the given objective and gradient.
     * The search stops when the bracket becomes sufficiently small or
     * the Wolfe conditions with given parameters are met.
     */
    slope_binary_search(const objective_fn& objective,
                        const gradient_fn& gradient,
                        const wolfe_param_type& wolfe_params,
                        const param_type& params = param_type())
      : f_(objective, gradient),
        wolfe_(boost::bind(&line_function<Vec>::value, &f_),
               boost::bind(&line_function<Vec>::slope, &f_),
               wolfe_params),
        params_(params) {
      assert(params.valid());
    }

    /**
     * Performs line search from x along the given direction.
     */
    real_type step(const Vec& x, const Vec& direction) {
      typedef typename line_function<Vec>::value_type value_type;

      // reset the function to the given line and initialize the Wolfe conds
      f_.reset(&x, &direction);
      wolfe_.reset();

      // make sure that the left derivative is < 0
      value_type left = f_.pos_slope(0.0);
      if (left.val > 0.0) {
        throw line_search_failed(
          "The function is increasing along the specified direction"
        );
      } else if (left.val == 0.0) {
        return left.pos;
      }

      // find the right bound s.t. the right derivative >= 0
      value_type right = f_.pos_slope(1.0);
      while (right.val < 0.0) {
        right = f_.pos_slope(right.pos * params_.multiplier);
        if (right.pos > params_.max_step) {
          throw line_search_failed(
            "Could not find right bound <= " + to_string(params_.max_step)
          );
        }
      }

      // perform binary search until we shrink the bracket sufficiently
      // and we have moved the left pointer
      while (right.pos - left.pos > params_.convergence || left.pos == 0.0) {
        value_type mid = f_.pos_slope((left + right) / 2.0);
        if (mid.val < 0.0) {
          left = mid;
        } else {
          right = mid;
        }
        if (mid.val == 0.0 || wolfe_(mid.pos)) {
          return mid.pos;
        }
        if (right.pos < params_.min_step) {
          throw line_search_failed("Step size is too small");
        }
      }

      // the left side of the bracket is guaranteed to have lower objective
      // than the start
      return left.pos;
    }

    // Private data
    //==========================================================================
  private:
    line_function<Vec> f_;
    wolfe_conditions<real_type> wolfe_;
    param_type params_;

  }; // class slope_binary_search

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
