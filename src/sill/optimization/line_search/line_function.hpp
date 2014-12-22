#ifndef SILL_LINE_FUNCTION_HPP
#define SILL_LINE_FUNCTION_HPP

#include <sill/optimization/line_search/line_search_result.hpp>

#include <boost/function.hpp>

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
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    /**
     * Creates a line function with the given base function and gradient.
     */
    explicit line_function(const objective_fn& objective = NULL,
                           const gradient_fn& gradient = NULL)
      : objective_(objective),
        gradient_(gradient),
        origin_(NULL),
        direction_(NULL),
        new_line_(true) { }

    /**
     * Sets the underlying objective and gradient.
     */
    void reset(const objective_fn& objective,
               const gradient_fn& gradient = NULL) {
      objective_ = objective;
      gradient_ = gradient;
      origin_ = NULL;
      direction_ = NULL;
      new_line_ = true;
    }
    
    /**
     * Selects the line to be restricted to in terms of the origin and
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
      return objective_(input_);
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
      return dot(*direction_, gradient_(input_));
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

    objective_fn objective_;
    gradient_fn gradient_;
    const Vec* origin_;
    const Vec* direction_;
    bool new_line_;
    real_type step_;
    Vec input_;

  }; // class line_function

} // namespace sill

#endif
