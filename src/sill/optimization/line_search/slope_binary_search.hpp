#ifndef SILL_SLOPE_BINARY_SEARCH_HPP
#define SILL_SLOPE_BINARY_SEARCH_HPP

#include <sill/global.hpp>
#include <sill/optimization/line_search/bracketing_line_search.hpp>
#include <sill/optimization/line_search/line_function.hpp>
#include <sill/optimization/line_search/line_search.hpp>
#include <sill/optimization/line_search/line_search_failed.hpp>
#include <sill/optimization/line_search/line_search_result.hpp>
#include <sill/optimization/line_search/wolfe_conditions.hpp>
#include <sill/parsers/string_functions.hpp>

#include <boost/bind.hpp>

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
    typedef line_search_result<real_type> result_type;
    typedef bracketing_line_search_parameters<real_type> param_type;
    typedef typename wolfe_conditions<real_type>::param_type wolfe_param_type;

    // Public functions
    //==========================================================================
  public:
    /**
     * Constructs the line search object.
     * The search stops when the bracket becomes sufficiently small.
     */
    explicit slope_binary_search(const param_type& params = param_type())
      : params_(params) {
      assert(params.valid());
    }

    /**
     * Constructs the line saerch object.
     * The search stops when the bracket becomes sufficiently small or
     * the Wolfe conditions with given parameters are met.
     */
    slope_binary_search(const param_type& params,
                        const wolfe_param_type& wolfe_params)
      : wolfe_(boost::bind(&line_function<Vec>::value, &f_, _1),
               boost::bind(&line_function<Vec>::slope, &f_, _1),
               wolfe_params),
        params_(params) {
      assert(params.valid());
    }

    void objective(gradient_objective<Vec>* obj) {
      f_.objective(obj);
    }

    result_type step(const Vec& x, const Vec& direction) {
      // reset the function to the given line and initialize the Wolfe conds
      f_.line(&x, &direction);
      wolfe_.reset();

      // make sure that the left derivative is < 0
      result_type left = f_.slope_result(0.0);
      if (left.value > 0.0) {
        throw line_search_failed(
          "The function is increasing along the specified direction"
        );
      } else if (left.value == 0.0) {
        return f_.value_result(left.step);
      }

      // find the right bound s.t. the right derivative >= 0
      result_type right = f_.slope_result(1.0);
      while (right.value < 0.0) {
        ++(this->bounding_steps_);
        right = f_.slope_result(right.step * params_.multiplier);
        if (right.step > params_.max_step) {
          throw line_search_failed(
            "Could not find right bound <= " + to_string(params_.max_step)
          );
        }
      }

      // perform binary search until we shrink the bracket sufficiently
      // and we have moved the left pointer
      while (right.step - left.step > params_.convergence || left.step == 0.0) {
        ++(this->selection_steps_);
        result_type mid = f_.slope_result((left.step + right.step) / 2);
        if (mid.value < 0.0) {
          left = mid;
        } else {
          right = mid;
        }
        if (mid.value == 0.0 || wolfe_(mid.step)) {
          return f_.value_result(mid.step);
        }
        if (right.step < params_.min_step) {
          throw line_search_failed("Step size is too small in selection");
        }
      }

      // the left side of the bracket is guaranteed to have lower objective
      // than the start
      return f_.value_result(left.step);
    }

    void print(std::ostream& out) const {
      out << "slope_binary_search(" << params_ << ", " << wolfe_.params() << ")";
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
