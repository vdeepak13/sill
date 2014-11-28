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
      real_type obj;
      value_type(real_type pos, real_type obj) : pos(pos), obj(obj) { }
    };
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    /**
     * Creates a line function with the given base function and gradient.
     */
    explicit line_function(const objective_fn& objective,
                           const gradient_fn& gradient = NULL)
      : objective_(objective), gradient_(gradient), origin_(NULL), direction_(NULL) { }
    
    /**
     * Selects the line to be restricted to in terms of the origin and
     * direction.
     */
    void line(const Vec* origin, const Vec* direction) {
      origin_ = origin;
      direction_ = direction;
    }

    /**
     * Evaluates the function at the given position and caches the result.
     */
    real_type operator()(real_type pos) {
      input_ = *direction_;
      input_ *= pos;
      input_ += *origin_;
      value_ = objective_(input_);
      return value_;
    }

    /**
     * Returns the value (i.e., position and objective) for the given position.
     */
    value_type value(real_type pos) {
      return value_type(pos, operator()(pos));
    }

    /**
     * Returns the input corresponding to the last invoked position.
     */
    const Vec& last_input() const {
      return input_;
    }

    /**
     * Returns the value corresponding ot the last invokd position.
     */
    real_type last_value() const {
      return value_;
    }

  private:
    objective_fn objective_;
    gradient_fn gradient_;
    const Vec* origin_;
    const Vec* direction_;
    Vec input_;

  }; // class line_function

} // namespace sill

#endif
