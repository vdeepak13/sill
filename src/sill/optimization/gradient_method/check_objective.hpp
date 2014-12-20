#ifndef SILL_CHECK_OBJECTIVE_HPP
#define SILL_CHECK_OBJECTIVE_HPP

#include <boost/function.hpp>

namespace sill {

  /**
   * A class that checks the validity of the objective and the associated
   * gradient function. The class verifies that the derivative along the
   * gradient is correctly approximated by the difference of the objective
   * function along the gradient. The argument eta specifies the accuracy
   * of the approximation.
   *
   * \tparam Vec a type that models the OptimizationVector concept.
   */
  template <typename Vec>
  class check_objective {
  public:
    typedef typename Vec::value_type real_type;
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    /**
     * Constructs the checker for the given objective function, gradient,
     * and a point at which the gradient is evaluated.
     */
    check_objective(objective_fn objective,
                    gradient_fn gradient,
                    const Vec& x)
      : objective_(objective), x_(x) {
      f_ = objective(x_);
      g_ = gradient(x_);
    }

    /**
     * Returns the difference between the exact derivative and the finite
     * element approximation. The difference should go to 0 as eta goes
     * to 0.
     */
    real_type operator()(real_type eta) const {
      return dot(g_, g_) - (objective_(x_ + g_ * eta) - f_) / eta;
    }
                         
  private:
    objective_fn objective_;
    real_type f_;
    Vec x_;
    Vec g_;

  }; // class check_objective

} // namespace sill

#endif
