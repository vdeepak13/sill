#ifndef SILL_LBFGS_HPP
#define SILL_LBFGS_HPP

#include <sill/math/constants.hpp>
#include <sill/optimization/concepts.hpp>
#include <sill/optimization/gradient_method/gradient_method.hpp>
#include <sill/optimization/line_search/line_search.hpp>

#include <boost/shared_ptr.hpp>

#include <vector>

namespace sill {

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
    typedef line_search_result<real_type> result_type;
    typedef boost::function<real_type(const Vec&)> objective_fn;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;
    
    struct param_type {
      /**
       * We declare convergence if the difference between the previous
       * and the new objective value is less than this threshold.
       */
      real_type convergence;

      /**
       * The number of previous gradients used for approximating the Hessian.
       */
      size_t history;
    
      param_type(real_type convergence = 1e-6, size_t history = 10)
        : convergence(convergence), history(history) { }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.convergence << " " << p.history;
        return out;
      }

    }; // struct param_type

    /**
     * Creates an L-BFGS minimizer using the given lin search algorithm
     * and parameters. The line_search object becomes owned by the lbfgs
     * object and will be deleted upon its destruction.
     */
    explicit lbfgs(line_search<Vec>* search,
                   const param_type& params = param_type())
      : search_(search),
        params_(params),
        shist_(params.history),
        yhist_(params.history),
        rhist_(params.history),
        iteration_(0),
        value_(nan()),
        converged_(false) { }

    void reset(const objective_fn& objective,
               const gradient_fn& gradient,
               const Vec& init) {
      search_->reset(objective, gradient);
      gradient_ = gradient;
      iteration_ = 0;
      x_ = init;
      value_ = nan();
      converged_ = false;
    }

    result_type iterate() {
      const Vec& g = gradient_(x_);
      
      // compute the direction
      if (iteration_ > 0) {
        dir_ = g;
        size_t m = std::min(params_.history, iteration_);
        std::vector<real_type> alpha(m + 1);
        for (size_t i = 1; i <= m; ++i) {
          alpha[i] = rho(i) * dot(s(i), dir_);
          dir_ -= alpha[i] * y(i);
        }
        for (size_t i = m; i >= 1; --i) {
          real_type beta = rho(i) * dot(y(i), dir_);
          dir_ += s(i) * (alpha[i] - beta);
        }
        dir_ *= -1.0;
      } else {
        dir_ = -g;
      }

      // update the solution and the historical values of s, y, and rho
      result_type result = search_->step(x_, dir_);
      size_t index = iteration_ % params_.history;
      shist_[index] = result.step * dir_;
      yhist_[index] = (iteration_ > 0) ? (g - g_) : g;
      rhist_[index] = real_type(1.0) / dot(shist_[index], yhist_[index]);
      x_ += shist_[index];
      g_ = g;
      ++iteration_;

      // determine the convergence
      converged_ = (value_ - result.value) < params_.convergence;
      value_ = result.value;
      return result;
    }

    bool converged() const {
      return converged_;
    }

    const Vec& solution() const {
      return x_;
    }

    void print(std::ostream& out) const {
      out << "lbfgs(" << params_ << ")";
    }

  private:
    //! Returns the i-th historical value of s, where i <= min(m, iteration_)
    const Vec& s(size_t i) const {
      return shist_[(iteration_ - i) % params_.history];
    }

    //! Returns the i-th historical value of y, where i <= min(m, iteration_)
    const Vec& y(size_t i) const {
      return yhist_[(iteration_ - i) % params_.history];
    }

    //! Returns the i-th historical value of rho, where i <= min(m, iteration_)
    real_type rho(size_t i) const {
      return rhist_[(iteration_ - i) % params_.history];
    }
    
    //! The gradient of the objective function
    gradient_fn gradient_;

    //! The line search algorithm
    boost::shared_ptr<line_search<Vec> > search_;

    //! Convergence and history parameters
    param_type params_;

    //! The history of solution differences x_{k+1} - x_k
    std::vector<Vec> shist_;

    //! The history of gradient differences g_{k+1} - g_k
    std::vector<Vec> yhist_;

    //! The history of rho_k = 1 / dot(y_k , s_k)
    std::vector<real_type> rhist_;

    //! Current iteration
    size_t iteration_;

    //! Last solution
    Vec x_;

    //! Last gradient
    Vec g_;

    //! Last direction
    Vec dir_;

    //! Last value
    real_type value_;

    //! True if the (last) iteration has converged
    bool converged_;

  }; // class lbfgs
  
} // namespace sill

#endif
