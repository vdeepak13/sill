
#ifndef SILL_GRADIENT_METHOD_HPP
#define SILL_GRADIENT_METHOD_HPP

#include <sill/base/stl_util.hpp>
#include <sill/math/is_finite.hpp>
#include <sill/optimization/basic_step_functor.hpp>
#include <sill/optimization/line_search_with_grad.hpp>
#include <sill/optimization/real_optimizer.hpp>
#include <sill/optimization/single_opt_step.hpp>
#include <sill/optimization/wolfe_step_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! Parameters for gradient_method subclasses.
  struct gradient_method_parameters {

    /**
     * Types of optimization steps (real_opt_step).
     *  - SINGLE_OPT_STEP: No line search.
     *  - LINE_SEARCH: Line search only using the objective.
     *       This is best if the gradient is expensive to compute.
     *  - LINE_SEARCH_WITH_GRAD: Line search using the objective and gradient.
     *       This is best if computing the gradient along with the objective
     *       is cheap.  In this case, it is probably a good idea to use
     *       approximate line search (i.e., early stopping).
     */
    enum real_opt_step_type {SINGLE_OPT_STEP, LINE_SEARCH,
                             LINE_SEARCH_WITH_GRAD};

    //! Optimization step type.
    //!  (default = LINE_SEARCH)
    real_opt_step_type step_type;

    //! Parameters for line search (for LINE_SEARCH, LINE_SEARCH_WITH_GRAD.
    line_search_parameters ls_params;

    //! Parameters for single opt steps (for SINGLE_OPT_STEP).
    single_opt_step_parameters single_opt_step_params;

    /**
     * Types of line search stopping conditions.
     * These are only relevant if using LINE_SEARCH_WITH_GRAD.
     *  - LS_EXACT: no early stopping
     *  - LS_STRONG_WOLFE: early stopping using strong Wolfe
     *                              conditions (fast and safe)
     *  - LS_WEAK_WOLFE: early stopping using weak Wolfe
     *                            conditions (fast and unsafe)
     */
    enum ls_stopping_type {LS_EXACT, LS_STRONG_WOLFE, LS_WEAK_WOLFE};

    //! Line search stopping condition (for LINE_SEARCH_WITH_GRAD).
    //!  (default = LS_EXACT)
    ls_stopping_type ls_stopping;

    //! Convergence tolerance.
    //! (default = .000001)
    double convergence_zero;

    /**
     * Debug mode:
     *  - 0: no debugging (default)
     *  - 1: print status through steps
     *  - 2: set line search debugging to true
     *  - 3: print out lots of info if this finds the function being
     *       optimized is non-convex
     *  - higher: revert to highest debugging mode
     */
    size_t debug;

    gradient_method_parameters()
      : step_type(LINE_SEARCH), ls_stopping(LS_EXACT),
        convergence_zero(.000001), debug(0) { }

    virtual ~gradient_method_parameters() { }

    bool valid(bool print_warnings = true) const;

  }; // struct gradient_method_parameters

  /**
   * Interface for gradient-based optimization algorithms.
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
  class gradient_method
    : public real_optimizer<OptVector> {

    concept_assert((sill::ObjectiveFunctor<Objective, OptVector>));
    concept_assert((sill::GradientFunctor<Gradient, OptVector>));

    // Public types
    //==========================================================================
  public:

    //! Base class
    typedef real_optimizer<OptVector> base;

    //! Options.
    typedef gradient_method_parameters parameters;

    // Protected types
    //==========================================================================
  protected:

    //! real_opt_step_functor type: basic line search
    typedef basic_step_functor<OptVector, Objective>
    basic_step_type;

    //! real_opt_step_functor type: Wolfe conditions
    typedef wolfe_step_functor<OptVector, Objective, Gradient>
    wolfe_step_type;

    /*
    //! Type for line_search_type = 0
    typedef line_search<line_search_objective_functor<OptVector,Objective> >
      line_search_type0;

    //! Type for line_search_type = 1
    typedef line_search<wolfe_conditions_functor<OptVector,Objective,Gradient>,
                        wolfe_conditions_functor<OptVector,Objective,Gradient> >
      line_search_type1;
    */

    // Protected data and methods
    //==========================================================================

    // Inherited from base class
    using base::x_;
    using base::objective_change_;
    using base::objective_;
    using base::iteration_;

    parameters params;

    //! From parameters:
    double convergence_zero;

    /*
    //! From parameters:
    parameters::real_opt_step_type step_type;

    //! From parameters:
    parameters::ls_stopping_type ls_stopping;

    //! From parameters:
    single_opt_step_parameters single_opt_step_params;

    //! From parameters:
    line_search_parameters line_search_params;

    //! From parameters:
    size_t debug;
    */

    //! Objective functor
    const Objective& obj_functor;

    //! Gradient functor
    const Gradient& grad_functor;

    //! Last update direction.
    OptVector direction_;

    //! Total number of objective function evaluations made in line search.
    size_t total_obj_calls_;

    //! Optimization step instance.
    real_opt_step* step_ptr;

    //! real_opt_step_functor instance: no line search
    void_step_functor* void_step_functor_ptr;

    //! real_opt_step_functor instance: basic line search
    basic_step_type* basic_step_type_ptr;

    //! real_opt_step_functor instance: Wolfe conditions
    wolfe_step_type* wolfe_step_type_ptr;

    // Protected methods
    //==========================================================================

    //! Free memory.
    void clear_pointers() {
      if (step_ptr)
        delete(step_ptr);
      step_ptr = NULL;
      if (void_step_functor_ptr)
        delete(void_step_functor_ptr);
      void_step_functor_ptr = NULL;
      if (basic_step_type_ptr)
        delete(basic_step_type_ptr);
      basic_step_type_ptr = NULL;
      if (wolfe_step_type_ptr)
        delete(wolfe_step_type_ptr);
      wolfe_step_type_ptr = NULL;
    }

    void init() {
      switch (params.step_type) {
      case parameters::SINGLE_OPT_STEP:
        step_ptr = new single_opt_step(params.single_opt_step_params);
        void_step_functor_ptr = new void_step_functor();
        break;
      case parameters::LINE_SEARCH:
        step_ptr = new line_search(params.ls_params);
        basic_step_type_ptr = new basic_step_type(obj_functor, x_, direction_);
        break;
      case parameters::LINE_SEARCH_WITH_GRAD:
        step_ptr = new line_search_with_grad(params.ls_params);
        wolfe_step_type_ptr = new wolfe_step_type(convergence_zero);
        switch (params.ls_stopping) {
        case parameters::LS_EXACT:
          wolfe_step_type_ptr->disable_early_stopping = true;
          break;
        case parameters::LS_STRONG_WOLFE:
          wolfe_step_type_ptr->disable_early_stopping = false;
          wolfe_step_type_ptr->use_strong_conditions = true;
          break;
        case parameters::LS_WEAK_WOLFE:
          wolfe_step_type_ptr->disable_early_stopping = false;
          wolfe_step_type_ptr->use_strong_conditions = false;
          break;
        default:
          assert(false);
        }
        break;
      default:
        assert(false);
      }

      /*
      case 0:
        switch(line_search_stopping) {
        case 0:
          init_ls0();
          break;
        case 1:
        case 2:
          init_ls1();
          break;
        default:
          assert(false);
        }
        break;
      case 1:
        init_ls1();
        switch(line_search_stopping) {
        case 0:
          ls_obj_functor1_ptr->disable_early_stopping = true;
          break;
        case 1:
        case 2:
          ls_obj_functor1_ptr->disable_early_stopping = false;
          break;
        default:
          assert(false);
        }
        break;
      default:
        assert(false);
      }
      */
    } // init()

    /**
     * Perform one line search, and update x_ unless at the optimum.
     * This assumes direction_ has already been computed.
     * @param  grad   Gradient at x_ (used for approximate line searches).
     * @return  False if at optimum.
     */
    bool run_line_search(const OptVector& grad) {
      if (params.debug > 0) {
        std::cerr << "gradient_method::run_line_search(): begin" << std::endl;
        if (params.debug > 2) {
          /*
            std::cerr << "gradient_method is about to run line search:\n"
                    << "\t from x = " << x_ << "\n"
                    << "\t in direction = " << direction_ << "\n";
          */
          // TO DO: FINISH PRINTING ALL OF THIS INFO.
        }
      }

      // Check step magnitude
      double step_magnitude(direction_.L2norm());
      if (!is_finite(step_magnitude)) {
        if (params.debug > 0)
          std::cerr << "gradient_method::run_line_search() failed since"
                    << " gradient was infinite." << std::endl;
        return false;
      }
      if (step_magnitude <= 0) { // do <= for numerical reasons
        if (params.debug > 0)
          std::cerr << "gradient_method::run_line_search() exited since"
                    << " gradient was 0."
                    << std::endl;
        return false;
      }

      /*
      if (params.debug > 0) {
        double gradnorm(grad.L2norm());
        std::cerr << "  Computed direction; gradient L2 norm = " << gradnorm
		  << std::endl;
      }
      */

      // Reset step functor, and run line search.
      assert(step_ptr);
      switch (params.step_type) {
      case parameters::SINGLE_OPT_STEP:
//          step_ptr->get_params().step_magnitude = step_magnitude; // TO DO
        step_ptr->step(*void_step_functor_ptr);
        break;
      case parameters::LINE_SEARCH:
        assert(basic_step_type_ptr);
        basic_step_type_ptr->reset(obj_functor, x_, direction_);
//          step_ptr->get_params().step_magnitude = step_magnitude; // TO DO
        step_ptr->step(*basic_step_type_ptr);
        break;
      case parameters::LINE_SEARCH_WITH_GRAD:
        assert(wolfe_step_type_ptr);
        {
          double direction_dot_grad(direction_.inner_prod(grad));
          if (direction_dot_grad >= 0) {
            if (params.debug > 0)
              std::cerr << "gradient_method::run_line_search() was given a"
                        << " direction which had a dot product of "
                        << direction_dot_grad << " with the gradient."
                        << std::endl;
            return false;
          }
          wolfe_step_type_ptr->reset(x_, direction_, obj_functor, grad_functor,
                                     objective_,
                                     direction_dot_grad, step_magnitude);
        }
//          step_ptr->get_params().step_magnitude = step_magnitude; // TO DO
        step_ptr->step(*wolfe_step_type_ptr);
        break;
      default:
        assert(false);
      }

      total_obj_calls_ += step_ptr->calls_to_objective();
      double eta(step_ptr->eta());
      x_ += direction_ * eta;
      double new_objective(step_ptr->valid_objective() ?
                           step_ptr->objective() :
                           obj_functor.objective(x_));

      /*
      if (step_magnitude * eta < convergence_zero) {
        if (step_magnitude * eta < -convergence_zero) {
          if (params.debug > 0)
            std::cerr << "gradient_method::run_line_search() did a line search"
                      << " and found the function being optimized is not"
                      << " convex (or you are having numerical issues)."
                      << " Set the debugging level higher to print out tons of"
                      << " info to demonstrate the non-convexity." << std::endl;
        }
        if (params.debug > 0)
          std::cerr << "gradient_method::run_line_search() exited step with"
                    << " eta = " << eta
                    << " < convergence_zero = " << convergence_zero << "\n"
                    << "  step direction L2 norm = " << direction_.L2norm()
                    << std::endl;
        return false;
      }
      */
      if (iteration_ != 0) {
        objective_change_ = new_objective - objective_;
        if (objective_change_ > - convergence_zero) {
          if (params.debug > 0) {
            if (objective_change_ > convergence_zero)
              std::cerr <<"gradient_method::run_line_search() did a line search"
                        << " and found the function being optimized is not"
                        << " convex (or you are having numerical issues). "
                        << "Set the debugging level higher to print out tons of"
                        << " info to demonstrate the non-convexity."
                        << std::endl;
            std::cerr << "gradient_method::run_line_search() exited step with "
                      << "objective_change_ = " << objective_change_
                      << " < convergence_zero = " << convergence_zero << "\n"
                      << "  step direction L2 norm = " << direction_.L2norm()
                      << std::endl;
          }
          return false;
        }
      }
      objective_ = new_objective;
      if (params.debug > 0) {
        double stepnorm(direction_.L2norm() * eta);
        std::cerr << " End of gradient_method::run_line_search():\n"
                  << "  step L2 norm = " << stepnorm
                  << ", new x L2 norm = " << x_.L2norm() << "\n"
                  << "  iteration = " << iteration_ << ", eta = " << eta
                  << ", objective = " << objective_ << ", objective change = "
                  << objective_change_ << std::endl;
      }
      ++iteration_;
      return true;
    } // run_line_search()

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor for gradient_method.
     * @param x_   Pre-allocated and initialized variables being optimized over.
     */
    gradient_method(const Objective& obj_functor, const Gradient& grad_functor,
                    OptVector& x_, const parameters& params)
      : base(x_, obj_functor.objective(x_)), params(params),
        convergence_zero(params.ls_params.convergence_zero),
        obj_functor(obj_functor), grad_functor(grad_functor),
        direction_(x_.size(), 0), total_obj_calls_(0),
        step_ptr(NULL), void_step_functor_ptr(NULL), basic_step_type_ptr(NULL),
        wolfe_step_type_ptr(NULL) {
      init();
    }

    virtual ~gradient_method() {
      clear_pointers();
    }

    //! Return the average number of objective function calls per line search,
    //! or -1 if no iterations have completed.
    double objective_calls_per_iteration() const {
      if (iteration_ == 0)
        return -1;
      else
        return ((double)(total_obj_calls_) / iteration_);
    }

  }; // class gradient_method

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_GRADIENT_METHOD_HPP
