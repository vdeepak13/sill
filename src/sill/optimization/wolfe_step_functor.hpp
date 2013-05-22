
#ifndef SILL_WOLFE_STEP_FUNCTOR_HPP
#define SILL_WOLFE_STEP_FUNCTOR_HPP

#include <sill/optimization/concepts.hpp>
#include <sill/optimization/real_opt_step_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor to be passed to line_search_with_grad by conjugate gradient or
   * other gradient/Hessian-based methods.
   * This computes whether or not the Wolfe conditions hold,
   * allowing early stopping.
   * This fits the LineSearchObjectiveFunctor and LineSearchGradientFunctor
   * concepts.
   *
   * About the Wolfe conditions:
   *  - The Wolfe conditions permit early stopping of a line search during
   *    optimization methods like conjugate gradient by showing that the line
   *    search in the current direction has made "enough" progress.
   *  - The weak Wolfe conditions are:
   *     - objective(x + eta * direction)
   *         <= objective(x) + c1 * eta * dot(direction, gradient(x))
   *     - dot(direction, gradient(x + eta * direction))
   *         >= c2 * dot(direction, gradient(x))
   *     - These do NOT give guarantees when used with optimization routines!
   *  - The strong Wolfe conditions are the same for the objective but change
   *    the gradient condition to:
   *     - | dot(direction, gradient(x + eta * direction)) |
   *         <= c2 * | dot(direction, gradient(x)) |
   *     - These give good guarantees when used with optimization routines for
   *       smooth functions.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   * @tparam Objective      Type of functor which computes the objective value.
   * @tparam Gradient       Type of functor which computes the gradient.
   *
   * @todo Organize this class' data better.
   *
   * \ingroup optimization_classes
   */
  template <typename OptVector, typename Objective, typename Gradient>
  struct wolfe_step_functor
    : public real_opt_step_functor {

    concept_assert((sill::OptimizationVector<OptVector>));
    concept_assert((sill::ObjectiveFunctor<Objective, OptVector>));
    concept_assert((sill::GradientFunctor<Gradient, OptVector>));

  private:

    //! Original x from which optimization is happening.
    const OptVector* x_ptr;

    //! Optimization direction.
    const OptVector* direction_ptr;

    //! Objective
    const Objective* obj_functor_ptr;

    //! Gradient
    const Gradient* grad_functor_ptr;

    //! Pre-computed value used in check:  objective(x)
    double objective_x;

    //! Pre-computed value used in check:  dot(direction(x), gradient(x))
    double direction_dot_grad;

    //! Pre-computed L2 norm of the optimization direction.
    double direction_L2norm;

    //! If true, then the last call to objective() indicated that line search
    //! can exit early.
    mutable bool stop_early_;

    //! Temp place to store (x + eta * direction).
    mutable OptVector tmp_x;

    //! Temp place to store gradient(x + eta * direction).
    mutable OptVector tmp_grad;

    //! Last eta for which the objective and gradient were computed.
    mutable double last_eta_;

    //! Last objective value computed: objective(last_eta_).
    mutable double last_objective_;

    //! Last gradient value computed: gradient(last_eta_).
    //! This equals direction(x).dot(gradient(x + eta * direction)).
    mutable double last_gradient_;

    //! Compute the objective and gradient as necessary.
    void compute_everything(double eta) const {
      if (eta == last_eta_)
        return;
      assert(x_ptr);
      assert(direction_ptr);
      assert(obj_functor_ptr);
      assert(grad_functor_ptr);
      tmp_x = *direction_ptr;
      tmp_x *= eta;
      tmp_x += (*x_ptr);
      last_objective_ = obj_functor_ptr->objective(tmp_x);
      grad_functor_ptr->gradient(tmp_grad, tmp_x);
      last_gradient_ = tmp_grad.dot(*direction_ptr);

      stop_early_ = false;
      if (!disable_early_stopping) {
        double obj_change_required(c1 * eta * direction_dot_grad);
        if ((obj_change_required < - convergence_zero) &&
            (last_objective_ <= objective_x + obj_change_required)) {
          if (use_strong_conditions) {
            if (fabs(last_gradient_) <= c2 * fabs(direction_dot_grad))
              stop_early_ = true;
          } else {
            if (last_gradient_ >= c2 * direction_dot_grad)
              stop_early_ = true;
          }
        }
      }
    } // compute_everything()

    wolfe_step_functor() { }

    // Public data (options)
    //==========================================================================
  public:

    /**
     * If true, use the strong Wolfe conditions (default);
     * if false, use the weak Wolfe conditions (which are unsafe).
     */
    bool use_strong_conditions;

    //! Objective constant for Wolfe conditions
    //! This must be in (0,1).
    //!  (default = .0001)
    double c1;

    //! Gradient constant for Wolfe conditions
    //! Smaller values make this closer to exact line search.
    //! This must be in [c1,1).
    //!  (default = .9)
    double c2;

    //! Convergence zero.
    //! For this to be valid, this MUST be set to the convergence_zero value
    //! being used by the caller.
    double convergence_zero;

    //! If set to false, then turn off early stopping. (default = false);
    bool disable_early_stopping;

    // Public methods
    //==========================================================================

    /**
     * Constructor.  With this constructor, reset() MUST be called before
     * doing a line search!
     */
    wolfe_step_functor(double convergence_zero)
      : x_ptr(NULL), direction_ptr(NULL),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL),
        objective_x(std::numeric_limits<double>::infinity()),
        direction_dot_grad(std::numeric_limits<double>::infinity()),
        direction_L2norm(-std::numeric_limits<double>::infinity()),
        stop_early_(false),
        last_eta_(-std::numeric_limits<double>::infinity()),
        last_objective_(std::numeric_limits<double>::infinity()),
        last_gradient_(std::numeric_limits<double>::infinity()),
        use_strong_conditions(true), c1(.0001), c2(.9),
        convergence_zero(convergence_zero), disable_early_stopping(false) {
    }

    /**
     * Constructor which readies the functor for use with a line search.
     * @param x          Base x from which line search starts.
     * @param direction  Optimization direction.
     */
    wolfe_step_functor(const OptVector& x, const OptVector& direction,
                       const Objective& obj_functor,
                       const Gradient& grad_functor,
                       double objective_x, double direction_dot_grad,
                       double direction_L2norm, double convergence_zero)
      : x_ptr(&x), direction_ptr(&direction),
        obj_functor_ptr(&obj_functor), grad_functor_ptr(&grad_functor),
        objective_x(objective_x), direction_dot_grad(direction_dot_grad),
        direction_L2norm(direction_L2norm),
        stop_early_(false), tmp_x(x.size(), 0.), tmp_grad(x.size(), 0.),
        last_eta_(-std::numeric_limits<double>::infinity()),
        last_objective_(std::numeric_limits<double>::infinity()),
        last_gradient_(std::numeric_limits<double>::infinity()),
        use_strong_conditions(true), c1(.0001), c2(.9),
        convergence_zero(convergence_zero), disable_early_stopping(false) {
      assert(direction_L2norm > 0);
    }

    /**
     * Method for resetting this for a different line search.
     */
    void reset(const OptVector& x, const OptVector& direction,
               const Objective& obj_functor, const Gradient& grad_functor,
               double objective_x, double direction_dot_grad,
               double direction_L2norm) {
      x_ptr = &x;
      direction_ptr = &direction;
      obj_functor_ptr = &obj_functor;
      grad_functor_ptr = &grad_functor;
      this->objective_x = objective_x;
      this->direction_dot_grad = direction_dot_grad;
      this->direction_L2norm = direction_L2norm;
      assert(direction_L2norm > 0);
      stop_early_ = false;
      if (tmp_x.size() != x.size()) {
        tmp_x.resize(x.size());
        tmp_grad.resize(x.size());
      }
      last_eta_ = -std::numeric_limits<double>::infinity();
      last_objective_ = std::numeric_limits<double>::infinity();
      last_gradient_ = std::numeric_limits<double>::infinity();
    }

    /**
     * Computes the value of the objective for step size eta,
     * i.e., objective(x + eta * direction).
     * This also checks if line search can exit early and sets the result
     * of stop_early().
     */
    double objective(double eta) const {
      compute_everything(eta);
      return last_objective_;
    }

    //! Returns true if the last call to objective() or gradient() recommended
    //! early stopping (for line search).
    bool stop_early() const {
      return stop_early_;
    }

    //! Computes the gradient of the objective (w.r.t. eta) for step size eta.
    double gradient(double eta) const {
      compute_everything(eta);
      return last_gradient_;
    }

    //! Returns true if the gradient() method has been implemented.
    bool valid_gradient() const {
      return true;
    }

  }; // struct wolfe_step_functor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_WOLFE_STEP_FUNCTOR_HPP
