#ifndef SILL_CONJUGATE_GRADIENT_HPP
#define SILL_CONJUGATE_GRADIENT_HPP

#include <sill/math/constants.hpp>
#include <sill/optimization/concepts.hpp>
#include <sill/optimization/gradient_method/gradient_method.hpp>
#include <sill/optimization/line_search/line_search.hpp>

#include <boost/math/special_functions.hpp>
#include <boost/shared_ptr.hpp>


namespace sill {

  /**
   * A class that performs (possibly preconditioned) conjugate
   * gradient descent to minimize an objective.
   *
   * \ingroup optimization_gradient
   *
   * \tparam Vec a type that satisfies the OptimizationVector concept
   */
  template <typename Vec>
  class conjugate_gradient : public gradient_method<Vec> {
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
       * The method for computing the update (beta). These methods
       * are equivalent for quadratic objectives, but differ for others.
       */
      enum update_method { FLETCHER_REEVES, POLAK_RIBIERE } update;
      
      /**
       * Ensures that beta is always >= 0.
       */
      bool auto_reset;
      
      param_type(real_type convergence = 1e-6,
                 update_method update = POLAK_RIBIERE,
                 bool auto_reset = true)
        : convergence(convergence),
          update(update),
          auto_reset(auto_reset) { }

      /**
       * Sets the update method according to the given string.
       */
      void parse_update(const std::string& str) {
        if (str == "fletcher_reeves") {
          update = FLETCHER_REEVES;
        } else if (str == "polak_ribiere") {
          update = POLAK_RIBIERE;
        } else {
          throw std::invalid_argument("Invalid update method");
        }
      }

      friend std::ostream& operator<<(std::ostream& out, const param_type& p) {
        out << p.convergence << " ";
        switch (p.update) {
        case FLETCHER_REEVES:
          out << "fletcher_reeves";
          break;
        case POLAK_RIBIERE:
          out << "polak_ribiere";
          break;
        default:
          out << "?";
          break;
        }
        out << " " << p.auto_reset;
        return out;
      }

    }; // struct param_type

    /**
     * Creates a conjugate_gradient object with the given line search
     * algorithm and convergence parameters. The line_seach object
     * becomes owned by this conjugate_gradient and will be deleted
     * upon destruction.
     */
    explicit conjugate_gradient(line_search<Vec>* search,
                                const param_type& params = param_type())
      : search_(search),
        params_(params),
        converged_(false),
        value_(nan()) { }

    void reset(const objective_fn& objective,
               const gradient_fn& gradient,
               const Vec& init) {
      search_->reset(objective, gradient);
      gradient_ = gradient;
      x_ = init;
      value_ = nan();
      converged_ = false;
    }

    void reset(const objective_fn& objective,
               const gradient_fn& gradient,
               const gradient_fn& precondg,
               const Vec& init) {
      search_->reset(objective, gradient);
      gradient_ = gradient;
      precondg_ = precondg;
      x_ = init;
      value_ = nan();
      converged_ = false;
    }

    result_type iterate() {
      if (precondg_) {
        direction_preconditioned();
      } else {
        direction_standard();
      }
      result_type result = search_->step(x_, dir_);
      x_ += result.step * dir_;
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
      out << "conjugate_gradient(" << params_ << ")";
    }

  private:
    /**
     * Performs one iteration of standard (not preconditioned)
     * conjugate gradient descent.
     */
    void direction_standard() {
      if (boost::math::isnan(value_)) {
        g_ = gradient_(x_);
        dir_ = -g_;
      } else {
        const Vec& g2 = gradient_(x_);
        dir_ *= beta(g_, g_, g2, g2);
        dir_ -= g2;
        g_ = g2;
      }
    }

    /**
     * Performs one iteration of preconditioned conjugate gradient
     * descent.
     */
    void direction_preconditioned() {
      if (boost::math::isnan(value_)) {
        g_ = gradient_(x_);
        p_ = precondg_(x_);
        dir_ = -p_;
      } else {
        const Vec& g2 = gradient_(x_);
        const Vec& p2 = precondg_(x_);
        dir_ *= beta(g_, p_, g2, p2);
        dir_ -= p2;
        g_ = g2;
        p_ = p2;
      }
    }

    /**
     * Computes the decay (beta).
     */
    real_type beta(const Vec& g, const Vec& p,
                   const Vec& g2, const Vec& p2) const {
      real_type value;
      switch (params_.update) {
      case param_type::FLETCHER_REEVES:
        value = dot(p2, g2) / dot(p, g);
        break;
      case param_type::POLAK_RIBIERE:
        value = (dot(p2, g2) - dot(p2, g)) / dot(p, g);
        break;
      default:
        throw std::invalid_argument("Unsupported update type");
      }
      return (params_.auto_reset && value < 0.0) ? 0.0 : value;
    }

    //! The line search algorithm
    boost::shared_ptr<line_search<Vec> > search_;

    //! The update and convergence parameters
    param_type params_;

    //! The gradient function
    gradient_fn gradient_;

    //! The preconditioned gradient function
    gradient_fn precondg_;

    //! Current solution
    Vec x_;

    //! Last gradient
    Vec g_;

    //! Last preconditioned gradient (when using precondigioning)
    Vec p_;

    //! Last descent direction
    Vec dir_;

    //! Last objective value
    real_type value_;

    //! True if the (last) iteration has converged
    bool converged_;

  }; // class conjugate_gradient

} // namespace sill

#endif