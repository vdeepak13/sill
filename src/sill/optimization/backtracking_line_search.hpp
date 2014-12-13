#ifndef SILL_BACKTRACKING_LINE_SEARCH_HPP
#define SILL_BACKTRACKING_LINE_SEARCH_HPP

#include <sill/optimization/opt_step.hpp>
#include <sill/serialization/serialize.hpp>

#include <boost/function.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for backtracking_line_search.
   * \ingroup optimization_algorithms
   */
  template <typename RealType>
  struct backtracking_line_search_parameters {

    /**
     * Acceptable decrease of the objective based on linear extrapolation.
     * Must be in (0, 0.5).
     */
    RealType acceptance;

    /**
     * Discount factor in (0, 1) by which step is shrunk during line search.
     */
    RealType discount;

    /**
     * If the step size reaches this value (>=0), the line search will throw 
     * an opt_step_too_small exception.
     */
    RealType min_step;

    /**
     * Constructs the parameters.
     */
    backtracking_line_search_parameters(RealType acceptance = 0.3,
                                        RealType discount = 0.7,
                                        RealType min_step = 1e-10)
      : acceptance(acceptance),
        discount(discount),
        min_step(min_step) {
      assert(valid());
    }

    /**
     * Returns true if the parameters are valid.
     */
    bool valid() const {
      return
        acceptance > 0.0 && acceptance < 0.5 &&
        discount > 0.0 && discount < 1.0 &&
        min_step >= 0.0;
    }

    /**
     * Serializes the parameters.
     */
    void save(oarchive& ar) const {
        ar << acceptance << discount << min_step;
    }

    /**
     * Deserializes the parameters.
     */
    void load(iarchive& ar) {
      ar >> acceptance >> discount >> min_step;
    }

    /**
     * Prints the parameters to the output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const backtracking_line_search_parameters& p) {
      out << p.acceptance << ' ' << p.discount << ' ' << p.min_step;
      return out;
    }
    
  }; // struct backtracking_line_search_parameters

  /**
   * A class that attempts to reduce the objective enough based on
   * a linear interpolation of the function. Given two parameters
   * stopping and discount, it will start with step size 1.0 and
   * reduce it by discount until the termination condition is met.
   *
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class backtracking_line_search : public opt_step<Vec> {
    // Public types
    //==========================================================================
  public:
    typedef typename Vec::value_type real_type;
    typedef backtracking_line_search_parameters<real_type> param_type;
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    // Public functions
    //==========================================================================
  public:
    backtracking_line_search(const objective_fn& objective,
                             const gradient_fn& gradient,
                             const param_type& params = param_type())
      : f_(objective),
        objective_(objective),
        gradient_(gradient),
        params_(params) {
      assert(params.valid());
    }

    real_type apply(Vec& x, const Vec& direction) {
      real_type threshold = params_.acceptance * dot(direction, gradient_(x));
      f_.line(&x, &direction);
      real_type f0 = objective_(x);
      real_type t = 1.0;
      while (t > params_.min_step && f_(t) > f0 + t * threshold) {
        t *= params_.discount;
      }
      if (t <= params_.min_step) {
        throw opt_step_too_small("Reached the minimum step size " + 
                                 to_string(params_.min_step));
      } else {
        x = f_.last_input();
        return f_.last_value();
      }
    }

    // Private data
    //==========================================================================
  private:
    line_function<Vec> f_;
    objective_fn objective_;
    gradient_fn gradient_;
    param_type params_;

  }; // class backtracking_line_search

} // namespace sill

#endif
