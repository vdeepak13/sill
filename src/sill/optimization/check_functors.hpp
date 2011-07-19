
#ifndef SILL_OPTIMIZATION_CHECK_FUNCTORS_HPP
#define SILL_OPTIMIZATION_CHECK_FUNCTORS_HPP

#include <sill/learning/validation/parameter_grid.hpp>
#include <sill/optimization/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Method for checking the validity of a GradientFunctor object
   * given an ObjectiveFunctor which is assumed to be correct.
   *
   * This does n iterations as follows:
   *  - Start from the given point x.
   *  - Compute the objective and gradient at x.
   *  - The gradient of objective(x + eta_k * gradient(x)) w.r.t. eta_k is
   *    gradient(x).L2norm()^2.
   *  - For k iterations,
   *     - Choose a step size eta_k, smaller each time.
   *     - Estimate the gradient via
   *        (objective(x + eta_k * gradient(x)) - objective(x)) / eta_k.
   *     - Compute the error: truth - estimate.
   *  - See whether or not the error decreases to 0 as eta_k goes to 0.
   *
   * The step sizes eta_k are chosen to lie between min_eta and max_eta on
   * a log scale.
   *
   * @param min_eta  Min eta to use.
   * @param max_eta  Max eta to use.
   * @param k        Number of steps.
   *
   * @return Vector of L2 errors of the estimates.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   * @tparam Gradient       Type of functor which computes the gradient.
   */
  template <typename OptVector, typename Objective, typename Gradient>
  vec check_objective_functor(const OptVector& x,
                              const Objective& obj_functor,
                              const Gradient& grad_functor,
                              double min_eta, double max_eta, size_t k) {
    concept_assert((sill::OptimizationVector<OptVector>));
    concept_assert((sill::ObjectiveFunctor<Objective, OptVector>));
    concept_assert((sill::GradientFunctor<Gradient, OptVector>));

    assert(min_eta > 0);
    assert(min_eta < max_eta);
    vec etas(create_parameter_grid(min_eta, max_eta, k, true));
    double obj_x(obj_functor.objective(x));
    OptVector grad_x;
    grad_functor.gradient(grad_x, x);
    double grad_x_wrt_eta = grad_x.L2norm();
    grad_x_wrt_eta *= grad_x_wrt_eta;
    vec errors = zeros(etas.size());
    OptVector x_;
    for (size_t i(0); i < k; ++i) {
      x_ = grad_x;
      x_ *= etas[i];
      x_ += x;
      double obj(obj_functor.objective(x_));
      errors[i] = grad_x_wrt_eta - (obj - obj_x) / etas[i];
    }
    return errors;
  }

  /*
   * THIS NEEDS TO BE FIXED!
   * Method for checking the validity of HessianDiagFunctor objects
   * given a GradientFunctor object which is assumed to be correct.
   *
   * This does n iterations as follows:
   *  - Start from the given point x.
   *  - Compute the gradient and the Hessian diagonal at x.
   *  - For k iterations,
   *     - Choose a step size eta_k, smaller each time.  We will look at the
   *       point (x + eta_k * gradient(x)).
   *     - Compute the true gradient(x + eta_k * gradient(x)).
   *     - Estimate this gradient via
   *        (gradient(x) + HessianDiag(x) * (eta_k * gradient(x))).
   *        - Note that this is most valid when the diagonal of the Hessian
   *          approximates the Hessian well.
   *     - Compute the L2 error of the estimate, divided by eta_k.
   *  - See whether or not the error decreases to 0 as eta_k goes to 0.
   *
   * The step sizes eta_k are chosen to lie between min_eta and max_eta on
   * a log scale.
   *
   * @param min_eta  Min eta to use.
   * @param max_eta  Max eta to use.
   * @param k        Number of steps.
   *
   * @return Vector of L2 errors of the estimates.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   * @tparam Gradient       Type of functor which computes the gradient.
   *
   * @todo Write a version of this which takes steps along single elements of x
   *       so that it is valid for any Hessians (not just diagonally-dominated
   *       ones).
   */
  /*
  template <typename OptVector, typename Gradient, typename HessianDiag>
  vec check_hessian_diag_functor(const OptVector& x,
                                 const Gradient& grad_functor,
                                 const HessianDiag& hd_functor,
                                 double min_eta, double max_eta, size_t k) {
    concept_assert((sill::OptimizationVector<OptVector>));
    concept_assert((sill::GradientFunctor<Gradient, OptVector>));
    concept_assert((sill::HessianDiagFunctor<HessianDiag, OptVector>));

    vec etas(create_parameter_grid(min_eta, max_eta, k, true));
    OptVector x_(x);
    OptVector grad_x;
    grad_functor.gradient(grad_x, x);
    OptVector hd_x_grad_x;
    hd_functor.hessian_diag(hd_x_grad_x, x);
    hd_x_grad_x.elem_mult(grad_x);
    vec errors(zeros<vec>(etas.size()));
    OptVector tmpoptvec;
    for (size_t i(0); i < k; ++i) {
      grad_functor(tmpoptvec, x + grad_x * etas[i]);
      tmpoptvec -= grad_x + hd_x_grad_x * etas[i];
      errors[i] = tmpoptvec.L2norm() / etas[i];
    }
    return errors;
  }
  */

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_OPTIMIZATION_CHECK_FUNCTORS_HPP
