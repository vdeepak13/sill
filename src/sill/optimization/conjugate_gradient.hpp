
#ifndef SILL_CONJUGATE_GRADIENT_HPP
#define SILL_CONJUGATE_GRADIENT_HPP

#include <sill/optimization/check_functors.hpp>
#include <sill/optimization/gradient_method.hpp>
#include <sill/optimization/void_preconditioner.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Parameters for conjugate gradient class.
  struct conjugate_gradient_parameters
    : public gradient_method_parameters {

    typedef gradient_method_parameters base;

    /**
     * Update method:
     *  - 0: beta = max{0, Polak-Ribiere}
     *    (default)
     */
    size_t update_method;

    conjugate_gradient_parameters()
      : base(), update_method(0) { }

    conjugate_gradient_parameters(const gradient_method_parameters& params)
      : base(params), update_method(0) { }

    bool valid() const {
      if (!base::valid())
        return false;
      if (update_method > 0)
        return false;
      return true;
    }

  }; // struct conjugate_gradient_parameters

  /**
   * Class for the conjugate gradient algorithm for unconstrained nonlinear
   * optimization.  This tries to minimize the given objective.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   * @tparam Objective      Type of functor which computes the objective value.
   * @tparam Gradient       Type of functor which computes the gradient.
   * @tparam Preconditioner Type of functor which applies a preconditioner.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  template <typename OptVector, typename Objective, typename Gradient,
            typename Preconditioner = void_preconditioner<OptVector> >
  class conjugate_gradient
    : public gradient_method<OptVector, Objective, Gradient> {

    concept_assert((sill::PreconditionerFunctor<Preconditioner, OptVector>));

    // Protected data and methods
    //==========================================================================
  protected:

    typedef gradient_method<OptVector, Objective, Gradient> base;

    // Import from base class:
    using base::grad_functor;
    using base::obj_functor;
    using base::x_;
    using base::params;
    using base::direction_;
    using base::iteration_;

    //! From parameters:
    size_t update_method;

    //! Preconditioner functor
    const Preconditioner* prec_functor_ptr;

    //! True if using preconditioning.
    bool using_preconditioning;

    //! Last gradient, and temporary to avoid reallocation (swapped).
    OptVector grad1;
    OptVector grad2;

    //! If true, then grad1 holds the last gradient.
    bool is_grad1;

    //! Same as grad1, grad2, but with the preconditioner applied.
    OptVector grad1prec;
    OptVector grad2prec;

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor for conjugate gradient without preconditioning.
     * @param x_   Pre-allocated and initialized variables being optimized over.
     */
    conjugate_gradient(const Objective& obj_functor,
                       const Gradient& grad_functor,
                       OptVector& x_,
                       const conjugate_gradient_parameters& params
                       = conjugate_gradient_parameters())
      : base(obj_functor, grad_functor, x_, params),
        update_method(params.update_method), prec_functor_ptr(NULL),
        using_preconditioning(false),
        grad1(x_.size(), 0), grad2(x_.size(), 0), is_grad1(false) {
    }

    /**
     * Constructor for conjugate gradient with preconditioning.
     * @param x_   Pre-allocated and initialized variables being optimized over.
     */
    conjugate_gradient(const Objective& obj_functor,
                       const Gradient& grad_functor,
                       const Preconditioner& prec_functor,
                       OptVector& x_,
                       const conjugate_gradient_parameters& params
                       = conjugate_gradient_parameters())
      : base(obj_functor, grad_functor, x_, params),
        update_method(params.update_method), prec_functor_ptr(&prec_functor),
        using_preconditioning(!boost::is_same<Preconditioner, void_preconditioner<OptVector> >::value),
        grad1(x_.size(), 0), grad2(x_.size(), 0), is_grad1(false) {
      if (using_preconditioning) {
        grad1prec.resize(x_.size());
        grad2prec.resize(x_.size());
      }
    }

    //! Perform one step.
    //! @return  False if at optimum.
    bool step() {
      if (params.debug > 1)
        std::cerr << "conjugate_gradient::step(): initial x L2 norm = "
                  << x_.L2norm() << std::endl;

      // Compute the direction
      if (iteration_ == 0) {
        grad_functor.gradient(grad1, x_);
        if (using_preconditioning) {
          grad1prec = grad1;
          prec_functor_ptr->precondition(grad1prec, x_);
          direction_ = grad1prec;
        } else {
          direction_ = grad1;
        }
        direction_ *= -1;
        is_grad1 = true;
      } else { // iteration_ > 0
        OptVector& last_grad = (is_grad1 ? grad1 : grad2);
        OptVector& new_grad = (is_grad1 ? grad2 : grad1);
        grad_functor.gradient(new_grad, x_);
        double beta(0);
        if (using_preconditioning) {
          OptVector& last_gradprec = (is_grad1 ? grad1prec : grad2prec);
          OptVector& new_gradprec = (is_grad1 ? grad2prec : grad1prec);
          new_gradprec = new_grad;
          prec_functor_ptr->precondition(new_gradprec, x_);
          switch(update_method) {
          case 0: // max{0, Polak-Ribiere}
            beta = last_grad.dot(last_gradprec);
            if (beta == 0)
              return false;
            beta = new_grad.dot(new_gradprec - last_gradprec) / beta;
            if (beta < 0)
              beta = 0;
            break;
          default:
            assert(false);
          }
          is_grad1 = !is_grad1;
          direction_ *= beta;
          direction_ -= new_gradprec;
        } else {
          switch(update_method) {
          case 0: // max{0, Polak-Ribiere}
            beta = last_grad.dot(last_grad);
            if (beta == 0)
              return false;
            beta = new_grad.dot(new_grad - last_grad) / beta;
            if (beta < 0)
              beta = 0;
            break;
          default:
            assert(false);
          }
          is_grad1 = !is_grad1;
          direction_ *= beta;
          direction_ -= new_grad;
        }
      }
      // Do a line search
      if (base::run_line_search(is_grad1 ? grad1 : grad2)) {
        return true;
      } else {
        if (params.debug > 2)
          std::cerr << "conjugate_gradient: resetting direction." << std::endl;
        /* If this was the first iteration, declare convergence.
           Otherwise, try resetting to the direction of steepest ascent.
           Try line search once more.
           If the line search says we have converged, then declare convergence.
         */
        if (iteration_ == 0)
          return false;
        OptVector& grad = (is_grad1 ? grad1 : grad2);
        if (using_preconditioning) {
          OptVector& gradprec = (is_grad1 ? grad1prec : grad2prec);
          gradprec = grad;
          prec_functor_ptr->precondition(gradprec, x_);
          direction_ = gradprec;
        } else {
          direction_ = grad;
        }
        direction_ *= -1;
        return base::run_line_search(is_grad1 ? grad1 : grad2);
      }
    } // step()

  }; // class conjugate_gradient

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CONJUGATE_GRADIENT_HPP
