#ifndef SILL_LINE_FUNCTION_HPP
#define SILL_LINE_FUNCTION_HPP

#include <sill/optimization/gradient_objective.hpp>
#include <sill/optimization/line_search/line_search_result.hpp>

namespace sill {

  /**
   * A class that represents a (possibly differentiable) function restricted
   * to a line. The original function (and, optionally, the gradient) is
   * specified at construction time. The line parameters are specified
   * using the line() member function.
   *
   * \tparam Vec a class that satisfies the OptimizationVector concept
   */
  template <typename Vec>
  class line_function {
  public:
    typedef typename Vec::value_type real_type;
    typedef line_search_result<real_type> result_type;

    /**
     * Creates a line function with the given base function and gradient.
     */
    explicit line_function(gradient_objective<Vec>* objective = NULL)
      : objective_(objective),
        origin_(NULL),
        direction_(NULL),
        new_line_(true) { }

    /**
     * Sets the objective that defines the value and gradient of
     * this function. The pointer is not owned by this object.
     */
    void objective(gradient_objective<Vec>* objective) {
      objective_ = objective;
    }
    
    /**
     * Sets the line to be restricted to in terms of the origin and
     * direction.
     */
    void line(const Vec* origin, const Vec* direction) {
      origin_ = origin;
      direction_ = direction;
      new_line_ = true;
    }

    /**
     * Evaluates the function for the given step size.
     */
    real_type value(real_type step) {
      cache_input(step);
      return objective_->value(input_);
    }

    /**
     * Returns the step-value pair for the given step size.
     */
    result_type value_result(real_type step) {
      return result_type(step, value(step));
    }

    /**
     * Returns the derivative (slope) at the given step size.
     */
    real_type slope(real_type step) {
      cache_input(step);
      return dot(*direction_, objective_->gradient(input_));
    }

    /**
     * Returns the step-slope pair for the given step size.
     */
    result_type slope_result(real_type step) {
      return result_type(step, slope(step));
    }

    /**
     * Returns the input for the last invocation to value() or slope().
     */
    const Vec& input() const {
      return input_;
    }

  private:
    /**
     * Updates the input for the given position if the position has changed.
     */
    void cache_input(real_type step) {
      if (new_line_ || step_ != step) {
        step_ = step;
        input_ = *origin_;
        if (step != 0.0) {
          input_ += *direction_ * step;
        }
        new_line_ = false;
      }
    }

    gradient_objective<Vec>* objective_;
    const Vec* origin_;
    const Vec* direction_;
    bool new_line_;
    real_type step_;
    Vec input_;

  }; // class line_function

} // namespace sill

#endif
