#ifndef SILL_LINE_FUNCTION_HPP
#define SILL_LINE_FUNCTION_HPP

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
    struct value_type {
      real_type pos;
      real_type val;
      pos_value(real_type pos, real_type val) : pos(pos), val(val) { }
    };
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    /**
     * Creates a line function with the given base function and gradient.
     */
    explicit line_function(const objective_fn& objective,
                           const gradient_fn& gradient = NULL)
      : objective_(objective),
        gradient_(gradient),
        origin_(NULL),
        direction_(NULL),
        new_line_(true) { }
    
    /**
     * Selects the line to be restricted to in terms of the origin and
     * direction.
     */
    void reset(const Vec* origin, const Vec* direction) {
      origin_ = origin;
      direction_ = direction;
      new_line_ = true;
    }

    /**
     * Evaluates the function at the given position.
     */
    real_type operator()(real_type pos) {
      return value(pos);
    }

    /**
     * Evaluates the function at the given position.
     */
    real_type value(real_type pos) {
      cache_input(pos);
      return objective_(input_);
    }

    /**
     * Returns the position-value pair for the given position.
     */
    value_type pos_value(real_type pos) {
      return value_type(pos, value(pos));
    }

    /**
     * Returns the derivative (slope) at the given position.
     */
    real_type slope()(real_type pos) {
      cache_input(pos);
      return dot(*direction, gradient_(input_));
    }

    /**
     * Returns the position-slope pair for the given position.
     */
    value_type pos_slope(real_type pos) {
      return value_type(pos, slope(pos));
    }

    /**
     * Returns the input corresponding to the last invoked position.
     */
    const Vec& input() const {
      return input_;
    }

  private:
    /**
     * Updates the input for the given position if the position has changed.
     */
    void cache_input(real_type pos) {
      if (new_line_ || pos_ != pos) {
        pos_ = pos;
        if (pos == 0.0) {
          input_ = *origin_;
        } else {
          input_ = *direction_;
          input_ *= pos;
          input_ += *origin_;
        }
        new_line_ = false;
      }
    }

    objective_fn objective_;
    gradient_fn gradient_;
    const Vec* origin_;
    const Vec* direction_;
    bool input_set_;
    real_type pos_;
    Vec input_;

  }; // class line_function

} // namespace sill

#endif
