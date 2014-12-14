#ifndef SILL_WOLFE_CONDITIONS_HPP
#define SILL_WOLFE_CONDITIONS_HPP

#include <boost/function.hpp>

namespace sill {

  /**
   * A class can evaluate whether or not Wolfe conditions hold for
   * the given point along the line search, permitting early stopping.
   * For more information, see http://en.wikipedia.org/wiki/Wolfe_conditions
   */
  template <typename RealType>
  class wolfe_conditions {
  public:
    typedef RealType real_type;

    /**
     * A type that represents a real-valued function (objective, gradient)
     * over the reals (position along the line).
     */
    typedef boost::function<real_type(real_type)> real_fn;

    /**
     * Parameters that govern the Wolfe conditions. Sensible defaults
     * will depend on the optimization algorithm used, see the static
     * functions. The defaults in the constructor effectively disable
     * early stopping.
     */
    struct param_type {
      real_type c1;
      real_type c2;
      bool strong;

      //! The defaults that fail the Wolfe conditions
      explicit param_type(real_type c1 = 1.0,
                          real_type c2 = 1.0,
                          bool strong = true)
        : c1(c1), c2(c2), strong(strong) { }

      //! The defaults for quasi-Newton methods
      static param_type quasi_newton(bool strong = true) {
        return param_type(1e-4, 0.9, strong);
      }
      
      //! The defaults for conjugate gradient descent
      static param_type conjugate_gradient(bool strong = true) {
        return param_type(1e-4, 0.1, strong);
      }

      //! Returns true if the parameters are valid
      bool valid() const {
        return 0.0 < c1 && c1 <= c2 && c2 <= 1.0;
      }
    }; // struct param_type

    /**
     * Constructs uninitialized Wolfe conditions that always fail.
     */
    wolfe_conditions() { }
    
    /**
     * Constructs Wolfe conditions for the given function and its
     * gradient. The member function reset() must be called before
     * the conditions can be evaluated.
     */
    wolfe_conditions(const real_fn& f,
                     const real_fn& g,
                     const param_type& params = param_type())
      : f_(f), g_(g), params_(params) {
      assert(params.valid());
    }

    /**
     * Initializes the starting position of the line search.
     */
    void reset() {
      if (f_ && g_) {
        f0_ = f_(0.0);
        g0_ = g_(0.0);
      }
    }
    
    /**
     * Evaluates the Wolfe conditions for the given step size.
     */
    bool operator()(real_type alpha) const {
      if (!f_) {
        return false;
      }
      if (params_.strong) {
        return
          f_(alpha) <= f0_ + params_.c1 * alpha * g0_ &&
          g_(alpha) >= params_.c2 * g0_;
      } else {
        return
          f_(alpha) <= f0_ + params_.c1 * alpha * g0_ &&
          std::fabs(g_(alpha)) <= params_.c2 * std::fabs(g0_);
      }
    }

  private:
    real_fn f_;
    real_fn g_;
    real_type f0_;
    real_type g0_;
    
  }; // class wolfe_conditions

} // namespace sill

#endif
