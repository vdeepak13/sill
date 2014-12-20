#ifndef SILL_LBFGS_HPP
#define SILL_LBFGS_HPP

#include <sill/global.hpp>
#include <sill/optimization/concepts.hpp>
#include <sill/optimization/gradient_method/gradient_method.hpp>
#include <sill/optimization/line_search/line_search.hpp>

#include <vector>

namespace sill {

  /**
   * Parameter struct for the L-BFGS class.
   */
  template <typename RealType>
  struct lbfgs_parameters {
    /**
     * The number of previous gradients used for approximating the Hessian.
     */
    size_t history;
    
    explicit lbfgs_parameters(size_t history = 10)
      : history(history) { }

    bool valid() const {
      return history > 0;
    }

  }; // class lbfgs_parameters


  /**
   * Class for the Limited Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
   * algorithm for unconstrained convex minimization.
   *
   * For more information, see, e.g.,
   *   D. C. Liu and J. Nocedal. On the Limited Memory Method for Large Scale
   *   Optimization (1989), Mathematical Programming B, 45, 3, pp. 503-528.
   *
   * \tparam Vec type that satisfies the OptimizationVector concept
   *
   * \ingroup optimization_gradient
   */
  template <typename Vec>
  class lbfgs : public gradient_method<Vec> {
  public:
    typedef typename Vec::value_type real_type;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;
    
    /**
     * Creates an L-BFGS minimizer with the given function and initial vector.
     * The line search object will become owned by this object and will be
     * deleted upon destruction.
     */
    lbfgs(objective_fn objective,
          gradient_fn gradient,
          line_search<Vec>* search,
          const Vec& init,
          const param_type& params = parm_type())
      : gradient_(gradient),
        line_search_(search),
        m_(param.history),
        shist_(param.history),
        yhist_(param.history),
        rhist_(param.history),
        iteration_(0),
        x_(init) {
      line_search_->init(objective, gradient);
    }

    bool iterate() {
      Vec g = gradient_(x_);
      
      // compute the direction
      if (iteration_ > 0) {
        dir_ = g;
        size_t m = std::min(m_, iteration_);
        std::vector<real_type> alpha(m + 1);
        for (size_t i = 1; i <= m; ++i) {
          alpha[i] = rho(i) * dot(s(i), dir);
          dir_ -= alpha[i] * y(i);
        }
        for (size_t i = m; i >= 1; --i) {
          real_type beta = rho(i) * dot(y(i), dir);
          dir_ += s(i) * (alpha[i] - beta);
        }
        dir_ *= -1.0;
      } else {
        dir_ = -g;
      }

      // update the solution and the historical values of s, y, and rho
      value_type step = line_search_->step(x_, dir_);
      size_t index = iteration_ % m_;
      shist_[index] = step.pos * dir_;
      yhist_[index] = g - g_;
      rhist_[index] = real_type(1.0) / dot(shist_[index], yhist_[index]);
      x_ += shist_[index];
      g_.swap(prevg_);
      ++iteration_;
    }

  private:
    //! Returns the i-th historical value of s, where i <= min(m_, iteration_)
    const Vec& s(size_t i) const {
      return shist_[(iteration_ - i) % m_];
    }

    //! Returns the i-th historical value of y, where i <= min(m_, iteration_)
    const Vec& y(size_t i) const {
      return yhist_[(iteration_ - i) % m_];
    }

    //! Returns the i-th historical value of rho, where i <= min(m_, iteration_)
    real_type rho(size_t i) const {
      return rhist_[(iteration_ - i) % m_];
    }
    
    //! The gradient of the objective function
    gradient_fn gradient_;

    //! The line search algorithm
    boost::unique_ptr<line_search<Vec> > line_search_;
    
    //! The window size
    size_t m_;

    //! The history of solution differences x_{k+1} - x_k
    std::vector<Vec> shist_;

    //! The history of gradient differences g_{k+1} - g_k
    std::vector<Vec> yhist_;

    //! The history of rho_k = 1 / dot(y_k , s_k)
    std::vector<real_type> rhist_;

    //! Current iteration
    size_t iteration_;

    //! The last solution
    Vec x_;

    //! The last gradient
    Vec g_;

    //! The last direction
    Vec dir_;

  }; // class lbfgs
  
} // namespace sill

#endif
