#ifndef SILL_GRADIENT_DESCENT_HPP
#define SILL_GRADIENT_DESCENT_HPP

#include <sill/optimization/concepts.hpp>
#include <sill/optimization/gradient_method/gradient_method.hpp>
#include <sill/optimization/line_search/line_search.hpp>

namespace sill {

  /**
   * A class the performs gradient descent to minimize an objective.
   * At each iteration, we perform line search to approximately
   * minimize the objective along the gradient direction.
   *
   * \ingroup optimization_gradient
   *
   * \tparam Vec a type that satisfies the OptimizationVector concept
   */
  template <typename Vec>
  class gradient_descent : public gradient_method<Vec> {
  public:
    typedef typename Vec::value_type real_type;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;
    
    /**
     * Creates a gradient_descent object with the given initial vector,
     * using the given line search algorithm. The line_search object
     * must not be used concurrently by other optimization objects.
     */
    gradient_descent(gradient_fn gradient,
                     const Vec& init,
                     line_search<Vec>* search)
      : x_(init), search_(search) { }

    bool iterate() {
      real_type last_val = step_.val;
      direction_ = -gradient_(x_);
      step_ = search_->step(x_, direction_);
      x_ += step_.pos * direction_;
      return (last_val - step_.val) < params_.convergence;
    }

    const Vec& x() const {
      return x_;
    }

    real_type objective() const {
      return step_.val;
    }

  private:
    //! The gradient function
    gradient_fn gradient_;

    //! The line search algorithm
    line_search<Vec>* search_;

    //! Current solution
    Vec x_;

    //! Last descent direction (-gradient)
    Vec direction_;

  }; // class gradient_descent

} // namespace sill

#endif
