
#ifndef PRL_LBFGS_HPP
#define PRL_LBFGS_HPP

#include <prl/optimization/gradient_method.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  //! Parameters for L-BFGS class.
  struct lbfgs_parameters
    : public gradient_method_parameters {

    typedef gradient_method_parameters base;

    //! Save M (> 0) previous gradients for estimating the Hessian.
    //!  (default = 10)
    size_t M;

    lbfgs_parameters()
      : base(), M(10) { }

    lbfgs_parameters(const gradient_method_parameters& params)
      : base(params), M(10) { }

    bool valid() const {
      if (!base::valid())
        return false;
      if (M == 0)
        return false;
      return true;
    }

  }; // struct lbfgs_parameters

  /**
   * Class for the Limited Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
   * algorithm for unconstrained convex optimization.
   * This tries to minimize the given objective.
   *
   * For more info on L-BFGS, see, e.g.,
   *   D. C. Liu and J. Nocedal. On the Limited Memory Method for Large Scale
   *   Optimization (1989), Mathematical Programming B, 45, 3, pp. 503-528.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   * @tparam Objective      Type of functor which computes the objective value.
   * @tparam Gradient       Type of functor which computes the gradient.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  template <typename OptVector, typename Objective, typename Gradient>
  class lbfgs
    : public gradient_method<OptVector, Objective, Gradient> {

    // Protected data and methods
    //==========================================================================
  protected:

    typedef gradient_method<OptVector, Objective, Gradient> base;

    // Import from base class:
    using base::grad_functor;
    using base::x_;
    using base::params;
    using base::direction_;
    using base::iteration_;

    size_t M_;

    //! Value of x before the last iteration.
    OptVector prev_x;

    //! Gradient(prev_x).
    OptVector prev_grad;

    //! Shist: x_{k+1} - x_k, from round k-M+1 to the most recent round k
    //! (inclusive so there are M elements total, or k after k iterations).
    //! I.e., the first element is: x_{iteration_+1} - x_{iteration_}.
    //! (Note: This is actually updated at the beginning of the next iteration.)
    std::list<OptVector*> Shist;

    //! Yhist: grad_{k+1} - grad_k.
    //! (See Shist.)
    std::list<OptVector*> Yhist;

    //! Holds rho_k.
    //! (See Shist.)
    std::list<double> rho_list;

    //! Temporary of size M to avoid reallocation.
    vec alphas;

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor for LBFGS.
     * @param x_   Pre-allocated and initialized variables being optimized over.
     */
    lbfgs(const Objective& obj_functor, const Gradient& grad_functor,
          OptVector& x_,
          const lbfgs_parameters& params = lbfgs_parameters())
      : base(obj_functor, grad_functor, x_, params),
        M_(params.M), alphas(M_, 0.) {
    }

    //! Perform one step.
    //! @return  False if at optimum.
    bool step() {
      if (params.debug > 1)
        std::cerr << "lbfgs::step(): initial x L2 norm = "
                  << x_.L2norm() << std::endl;
      // Compute the gradient, and update Shist, Yhist.
      grad_functor.gradient(direction_, x_);
      if (iteration_ != 0) {
        if (Shist.size() >= M_) {
          // Shift the front vectors to the back.
          Shist.push_back(Shist.front());
          Shist.pop_front();
          Yhist.push_back(Yhist.front());
          Yhist.pop_front();
          Shist.back()->operator=(x_);
          Yhist.back()->operator=(direction_);
          rho_list.pop_front();
        } else {
          // Make a place to store s_k, y_k.
          Shist.push_back(new OptVector(x_));
          Yhist.push_back(new OptVector(direction_));
        }
        Shist.back()->operator-=(prev_x);
        Yhist.back()->operator-=(prev_grad);
        rho_list.push_back(1. / Shist.back()->inner_prod(*(Yhist.back())));
      }
      prev_x = x_;
      prev_grad = direction_;

      // Compute the direction
      size_t bound(Shist.size());
      typename std::list<OptVector*>::const_reverse_iterator
        Shist_rit(Shist.rbegin());
      typename std::list<OptVector*>::const_reverse_iterator
        Yhist_rit(Yhist.rbegin());
      typename std::list<double>::const_reverse_iterator
        rho_list_rit(rho_list.rbegin());
      for (size_t i(0); i < bound; ++i) {
        const OptVector& s = *(*Shist_rit);
        const OptVector& y = *(*Yhist_rit);
        alphas[bound-i-1] = (*rho_list_rit) * direction_.inner_prod(s);
        direction_ -= y * alphas[bound-i-1]; //COULD AVOID ALLOCATION
        ++Shist_rit;
        ++Yhist_rit;
        ++rho_list_rit;
      }
      // direction_ = H_0 * direction_ // Could do this later.
      typename std::list<OptVector*>::const_iterator Shist_it(Shist.begin());
      typename std::list<OptVector*>::const_iterator Yhist_it(Yhist.begin());
      typename std::list<double>::const_iterator rho_list_it(rho_list.begin());
      for (size_t i(0); i < bound; ++i) {
        const OptVector& s = *(*Shist_it);
        const OptVector& y = *(*Yhist_it);
        direction_ +=
          s * (alphas[i] - (*rho_list_it) * direction_.inner_prod(y));
        ++Shist_it;
        ++Yhist_it;
        ++rho_list_it;
      }
      direction_ *= -1;

      // Do a line search
      if (base::run_line_search(prev_grad)) {
        return true;
      } else {
        if (params.debug > 2)
          std::cerr << "lbfgs: resetting direction." << std::endl;
        /* If Shist,Yhist are empty (i.e., the direction did not use the
           estimate of the Hessian), then declare convergence.
           Otherwise, try resetting Shist, Yhist.  Try line search once more.
           If the line search says we have converged, then declare convergence.
        */
        if (Shist.size() == 0)
          return false;
        foreach(OptVector* v_ptr, Shist)
          delete(v_ptr);
        foreach(OptVector* v_ptr, Yhist)
          delete(v_ptr);
        Shist.clear();
        Yhist.clear();
        rho_list.clear();
        direction_ = prev_grad;
        direction_ *= -1;
        return base::run_line_search(prev_grad);
      }
    } // step()

  }; // class lbfgs

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LBFGS_HPP
