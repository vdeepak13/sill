#ifndef SILL_VALUE_BINARY_SEARCH_HPP
#define SILL_VALUE_BINARY_SEARCH_HPP

#include <sill/global.hpp>
#include <sill/optimization/line_search/bracketing_line_search.hpp>
#include <sill/optimization/line_search/line_function.hpp>
#include <sill/optimization/line_search/line_search.hpp>
#include <sill/optimization/line_search/line_search_failed.hpp>
#include <sill/optimization/line_search/line_search_result.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class that performs line search by evaluating the objective value only.
   * This approach is preferable when evaluating the objective is much cheaper
   * than the gradient. For each invocation of step(), we maintain three step
   * sizes, left, mid, and right, such that f(mid) < f(0), f(mid) < f(left),
   * f(mid) < f(right). We declare convergence once |right-left| becomes
   * sufficiently small, returning mid as the result.
   *
   * This class works for both convex and non-convex, multi-modal objectives
   * and is guaranteed to decrease the objective value.
   *
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class value_binary_search : public line_search<Vec> {

    // Public types
    //==========================================================================
  public:
    typedef typename Vec::value_type real_type;
    typedef line_search_result<real_type> result_type;
    typedef bracketing_line_search_parameters<real_type> param_type;

    // Public functions
    //==========================================================================
  public:
    /**
     * Constructs an object that performs line search with objective
     * function alone.
     */
    explicit value_binary_search(const param_type& params = param_type())
      : params_(params) {
      assert(params.valid());
    }

    void objective(gradient_objective<Vec>* obj) {
      f_.objective(obj);
    }
    
    result_type step(const Vec& x, const Vec& direction) {
      // set the line
      f_.line(&x, &direction);
      result_type left  = f_.value_result(0.0);
      result_type mid   = f_.value_result(1.0);
      result_type right = mid;
      
      // identify the initial bracket
      if (right.value > left.value) { // shrink mid until its objective is < left
        mid = f_.value_result(1.0 / params_.multiplier);
        while (mid.value > left.value) {
          ++(this->bounding_steps_);
          right = mid;
          mid = f_.value_result(mid.step / params_.multiplier);
          if (right.step < params_.min_step) {
            throw line_search_failed("Step size too small in bounding");
          }
        }
      } else { // increase right until its objective is > mid
        right = f_.value_result(params_.multiplier);
        while (right.value < mid.value) {
          ++(this->bounding_steps_);
          left = mid;
          mid = right;
          right = f_.value_result(right.step * params_.multiplier);
          if (right.step > params_.max_step) {
            throw line_search_failed("Step size too large in bounding");
          }
        }
      }

      // do binary search while maintaining the invariant
      while (right.step - left.step > params_.convergence ||
             left.step == 0.0) {
        ++(this->selection_steps_);
        result_type mid_left = f_.value_result((left.step + mid.step) / 2);
        result_type mid_right = f_.value_result((mid.step + right.step) / 2);
        if (mid_left.value > mid.value && mid_right.value > mid.value) {
          left = mid_left;
          right = mid_right;
        } else if (mid_left.value < mid_right.value) {
          right = mid;
          mid = mid_left;
        } else {
          left = mid;
          mid = mid_right;
        }
        if (right.step < params_.min_step) {
          throw line_search_failed("Step size too small in selection");
        }
      }

      return mid;
    }

    void print(std::ostream& out) const {
      out << "value_binary_search(" << params_ << ")";
    }

  private:
    line_function<Vec> f_;
    param_type params_;

  }; // class value_binary_search

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_OPTIMIZATION_LINE_SEARCH_HPP
