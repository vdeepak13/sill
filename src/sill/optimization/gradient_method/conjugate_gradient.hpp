#ifndef SILL_CONJUGATE_GRADIENT_HPP
#define SILL_CONJUGATE_GRADIENT_HPP

#include <sill/optimization/concepts.hpp>
#include <sill/optimization/gradient_method/gradient_method.hpp>
#include <sill/optimization/line_search/line_search.hpp>

namespace sill {

  /**
   * A class that performs conjugate gradient descent to minimize
   * an objective.
   *
   * \ingroup optimization_gradient
   * \tparam Vec a type that satisfies the OptimizationVector concept
   */
  template <typename Vec>
  class conjugate_gradient : public gradient_method<Vec> {
  public:
    typedef typename Vec::value_type real_type;
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    /**
     * Creates a conjugate_gradient object with the given initial vector,
     * using the given line search algorithm. The line_search object
     * must not be used concurrently by other optimization objects.
     */
    conjugate_gradient(objective_fn objective,
                       gradient_fn gradient,
                       line_search<Vec>* search,
                       const Vec& init)
      : gradient_(gradient), x_(init) { }

    bool iterate() {
      if (precondition_) {
        direction_preconditioned();
      } else {
        direction_standard();
      }
      step_ = line_search_->step(x_, dir_);
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
    /**
     * Performs one iteration of standard (not preconditioned)
     * conjugate gradient descent.
     */
    void direction_standard() {
      if (iteration_ == 0) {
        neg_ = -g_(x_);
        dir_ = neg_;
      } else {
        Vec neg2 = -g_(x_);
        dir_ *= beta(neg2, neg_);
        dir_ += neg2;
        neg_.swap(neg2);
      }
    }

    /**
     * Performs one iteration of preconditioned conjugate gradient
     * descent.
     */
    void direction_preconditioned() {
      if (iteration_ == 0) {
        neg_ = -g_(x_);
        precondition_(x_, neg_, pre_);
        dir_ = pre_;
      } else {
        Vec neg2 = -g_(x_);
        Vec pre2;
        precondition_(x_, neg2, pre2);
        dir_ *= beta(pre2, pre_);
        dir_ += pre2;
        neg_.swap(neg2);
        pre_.swap(pre2);
      }
    }

    /**
     * Computes the beta.
     */
    real_type beta(const Vec& neg2, const Vec& pre2, const Vec& pre) const {
      real_type value;
      switch (params_.update) {
      case param_type::FLETCHER_REEVES:
        value = dot(pre2, neg2) / dot(pre, neg_);
        break;
      case param_type::POLAK_RIBIERE:
        value = dot(pre2, neg2 - neg_) / dot(pre, neg_);
        break;
      default:
        throw std::invalid_argument("Unsupported update type");
      }
      return (params_.auto_reset && value < 0.0) ? 0.0 : value;
    }

    //! The objective function
    objective_fn f_;
    
    //! The gradient function
    gradient_fn g_;

    //! The preconditioner
    preconditioner_fn precondition_;

    //! The line search algorithm
    line_search<Vec>* search_;

    //! Current solution
    Vec x_;

    //! Last negative gradient
    Vec neg_;

    //! Last preconditioned gradient (used if precondition_ is not null)
    Vec pre_;

    //! Last descent direction
    Vec dir_;

  }; // class conjugate_gradient

} // namespace sill

#endif
