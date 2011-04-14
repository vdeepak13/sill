#ifndef SILL_CRF_PARAMETER_LEARNER_HPP
#define SILL_CRF_PARAMETER_LEARNER_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/timer.hpp>

#include <sill/factor/concepts.hpp>
#include <sill/learning/crf/crf_parameter_learner_parameters.hpp>
#include <sill/learning/crf/crf_validation_functor.hpp>
#include <sill/learning/crossval_methods.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/vector_assignment_dataset.hpp>
#include <sill/learning/validation/validation_framework.hpp>
#include <sill/math/permutations.hpp>
#include <sill/math/is_finite.hpp>
#include <sill/math/statistics.hpp>
#include <sill/model/crf_model.hpp>
#include <sill/optimization/conjugate_gradient.hpp>
#include <sill/optimization/gradient_descent.hpp>
#include <sill/optimization/lbfgs.hpp>
#include <sill/optimization/stochastic_gradient.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  template <typename F> class crf_validation_functor;

  /**
   * This is a class which represents algorithms for learning a crf_model
   * from data.  This treats models as a collection of factors with these
   * requirements:
   *  - Instantiating the factors with fixed x (input) values should produce
   *    a LOW TREE-WIDTH decomposable model.
   *  - The first derivative (and ideally higher-order derivatives) of a
   *    factor can be computed.
   * See the LearnableCRFfactor concept for details about the requirements.
   * 
   * @tparam F  factor/potential type which implements the LearnableCRFfactor
   *            concept
   *
   * @see crf_model, crf_parameter_learner_parameters,
   *      crf_parameter_learner_builder
   * \ingroup learning_param
   */
  template <typename F>
  class crf_parameter_learner {

//    concept_assert((LearnableCRFfactor<F>));

    // Public type declarations
    // =========================================================================
  public:

    typedef dense_linear_algebra<> la_type;

    typedef record<la_type> record_type;

    //! The type of potentials/factors.
    typedef F crf_factor;

    //! The CRF model type.
    typedef crf_model<crf_factor> crf_model_type;

    //! CRF factor weights for optimization.
    typedef typename crf_model_type::opt_variables opt_variables;

    //! CRF factor regularization type.
    typedef typename crf_factor::regularization_type crf_factor_reg_type;

    //! Learning options.
    typedef crf_parameter_learner_parameters parameters;

    //! Functor used for cross validation to choose lambda;
    //! computes the score for a single record.
    //! This score is meant to be minimized.
    struct cross_val_functor {

      const crf_model_type& crf_;

      size_t score_type;

      cross_val_functor(const crf_model_type& crf_, size_t score_type)
        : crf_(crf_), score_type(score_type) { }

      double operator()(const record_type& r) const {
        switch(score_type) {
        case 0: // log likelihood
          return - crf_.log_likelihood(r);
        case 1: // per-label accuracy
          return - crf_.per_label_accuracy(r);
        case 2: // all-or-nothing label accuracy
          return - crf_.accuracy(r);
        case 3: // mean squared error
                // (\propto per-label accuracy for finite data)
          return crf_.mean_squared_error(r);
        default:
          assert(false);
          return std::numeric_limits<double>::infinity();
        }
      }

    }; // struct cross_val_functor

    // Private types
    // =========================================================================
  private:

    /**
     * A combination of:
     *  - Objective functor (fitting the ObjectiveFunctor concept).
     *  - Gradient functor (fitting the GradientFunctor concept).
     *  - Functor for applying the diagonal of the Hessian as a preconditioner
     *     (fitting the PreconditionerFunctor concept).
     * All of these functor types share the computation of conditioning the CRF
     * and can be sped up via this combined functor. This shared computation
     * should be done differently for different optimization methods:
     *  - For params.gm_params.step_type == LINE_SEARCH,
     *     - If computing the objective, only compute the objective.
     *     - If computing the gradient, also compute
     *        - the objective (always)
     *        - the preconditioner (if using preconditioning)
     *     - If computing the preconditioner, compute everything.
     *  - For params.gm_params.step_type == LINE_SEARCH_WITH_GRAD,
     *     - If computing the objective, compute the gradient too.
     *     - If computing the gradient, also compute
     *        - the objective (always)
     *        - the preconditioner (if using preconditioning) (since objective
     *          is always called before gradient by the line search methods)
     *     - If computing the preconditioner, compute everything.
     */
    class everything_functor {

      const crf_parameter_learner& cpl;

      //! If true, use shared computation.
      bool no_shared_computation;

      //! x for which the objective/gradient/Hessian diagonal might have been
      //! pre-computed.
      mutable opt_variables current_x;

      /**
       * Bit used by shared computation:
       *  - If true, then objective() should check to see if x == current_x
       *    (in which case the stored values may be used).
       *  - If false, then objective() does not need to check.
       */
      mutable bool obj_check_current_x;

      //! Stored objective value.
      mutable double current_objective;

      /**
       * Bit used by shared computation:
       *  - If true, then gradient() should check to see if x == current_x
       *    (in which case the stored values may be used).
       *  - If false, then gradient() does not need to check.
       */
      mutable bool grad_check_current_x;

      //! Stored gradient value.
      mutable opt_variables current_gradient;

      /**
       * Bit used by shared computation:
       *  - If true, then precondition() should check to see if x == current_x
       *    (in which case the stored values may be used).
       *  - If false, then precondition() does not need to check.
       */
      mutable bool hd_check_current_x;

      //! Stores the Hessian diagonal when needed.
      //! This is used iff cpl.params specifies this preconditioner.
      mutable opt_variables hd;

    public:
      /**
       * Constructor.
       */
      everything_functor(const crf_parameter_learner& cpl,
                         bool no_shared_computation)
        : cpl(cpl), no_shared_computation(no_shared_computation),
          obj_check_current_x(false),
          current_objective(std::numeric_limits<double>::infinity()),
          grad_check_current_x(false), hd_check_current_x(false) {
        if (!no_shared_computation) {
          current_x.resize(cpl.crf_.weights().size());
          current_gradient.resize(cpl.crf_.weights().size());
        }
        if (cpl.params.opt_method ==
            real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC) {
          hd.resize(cpl.crf_.weights().size());
        }
      }

      //! Computes the value of the objective at x.
      double objective(const opt_variables& x) const {
        try {
          if (!no_shared_computation) {
            if (obj_check_current_x && (x == current_x)) {
              return current_objective;
            } else {
              switch(cpl.params.gm_params.step_type) {
              case gradient_method_parameters::SINGLE_OPT_STEP:
                assert(false);
                break;
              case gradient_method_parameters::LINE_SEARCH:
                current_objective = cpl.my_objective(x);
                current_x = x;
                obj_check_current_x = true;
                grad_check_current_x = false;
                hd_check_current_x = false;
                break;
              case gradient_method_parameters::LINE_SEARCH_WITH_GRAD:
                cpl.my_everything(current_objective, current_gradient, hd, x,1);
                current_x = x;
                obj_check_current_x = true;
                grad_check_current_x = true;
                hd_check_current_x = false;
                break;
              default:
                assert(false);
              }
              return current_objective;
            }
          } else {
            return cpl.my_objective(x);
          }
        } catch (normalization_error exc) {
          throw normalization_error((std::string("crf_parameter_learner::everything_functor::objective() could not normalize the CRF; consider using more regularization (Message from normalization attempt: ") + exc.what() + ")").c_str());
        }
      }

      //! Computes the gradient of the function at x.
      //! @param grad  Data type in which to store the gradient.
      void gradient(opt_variables& grad, const opt_variables& x) const {
        try {
          if (!no_shared_computation) {
            if (grad_check_current_x && (x == current_x)) {
              grad = current_gradient;
            } else {
              if (cpl.params.opt_method ==
                  real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC) {
                cpl.my_everything(current_objective, current_gradient, hd, x,0);
                hd.reciprocal();
                hd_check_current_x = true;
              } else {
                cpl.my_everything(current_objective, current_gradient, hd, x,1);
                hd_check_current_x = false;
              }
              current_x = x;
              obj_check_current_x = true;
              grad_check_current_x = true;
              grad = current_gradient;
            }
          } else {
            cpl.my_gradient(grad, x);
          }
        } catch (normalization_error exc) {
          throw normalization_error((std::string("crf_parameter_learner::everything_functor::gradient() could not normalize the CRF; consider using more regularization (Message from normalization attempt: ") + exc.what() + ")").c_str());
        }
      }

      //! Applies the preconditioner to the given direction,
      //! when the optimization variables have value x.
      void
      precondition(opt_variables& direction, const opt_variables& x) const {
        try {
          if (!no_shared_computation) {
            if (hd_check_current_x && (x == current_x)) {
              direction.elem_mult(hd);
            } else {
              cpl.my_everything(current_objective, current_gradient, hd, x, 0);
              hd.reciprocal();
              current_x = x;
              obj_check_current_x = true;
              grad_check_current_x = true;
              hd_check_current_x = true;
              direction.elem_mult(hd);
            }
          } else {
            cpl.my_hessian_diag(hd, x);
            hd.reciprocal();
            direction.elem_mult(hd);
          }
        } catch (normalization_error exc) {
          throw normalization_error((std::string("crf_parameter_learner::everything_functor::precondition() could not normalize the CRF; consider using more regularization (Message from normalization attempt: ") + exc.what() + ")").c_str());
        }
      }

      //! Applies the last computed preconditioner to the given direction.
      void precondition(opt_variables& direction) const {
        direction.elem_mult(hd);
      }

    }; // class everything_functor

    /**
     * Version of everything_functor which is made for stochastic gradient.
     * It currently computes the gradient and objective for a datapoint
     * chosen uniformly at random.
     */
    class stochastic_everything_functor {

      const crf_parameter_learner& cpl;

      //! Objective value for the datapoint used to compute the last gradient.
      mutable double last_objective_;

    public:
      //! Constructor.
      stochastic_everything_functor(const crf_parameter_learner& cpl)
        : cpl(cpl) { }

      //! Objective value for the datapoint used to compute the last gradient.
      double last_objective() const {
        return last_objective_;
      }

      //! Computes the gradient at x by sampling a single datapoint.
      //! @param grad  Data type in which to store the gradient.
      void gradient(opt_variables& grad, const opt_variables& x) const {
        try {
          last_objective_ = cpl.my_stochastic_gradient(grad, x);
        } catch (normalization_error exc) {
          throw normalization_error
            (std::string("crf_parameter_learner::stochastic_everything_functor")
             + "::gradient() could not normalize the CRF; consider using more"
             + " regularization (Message from normalization attempt: "
             + exc.what() + ")");
        }
      }

    }; // stochastic_everything_functor

    //! Types for optimization methods
    typedef
    gradient_method<opt_variables,everything_functor,everything_functor>
    gradient_method_type;

    typedef
    gradient_descent<opt_variables,everything_functor,everything_functor>
    gradient_descent_type;

    typedef
    conjugate_gradient<opt_variables,everything_functor,everything_functor>
    conjugate_gradient_type;

    typedef conjugate_gradient<opt_variables,everything_functor,
                               everything_functor,everything_functor>
    prec_conjugate_gradient_type;

    typedef lbfgs<opt_variables,everything_functor,everything_functor>
    lbfgs_type;

    typedef stochastic_gradient<opt_variables,stochastic_everything_functor>
    stochastic_gradient_type;

    // Private data members
    // =========================================================================

    parameters params;

    //! Copied from parameters (and checked).
    typename crf_factor::regularization_type regularization;

    //! Training dataset
    const dataset<la_type>& ds;

    //! Iterator for training dataset
    mutable typename dataset<la_type>::record_iterator ds_it;

    //! Iterator at end of training dataset
    typename dataset<la_type>::record_iterator ds_end;

    //! The underlying CRF model.
    //! This is mutable to allow for evaluation of new weight values during
    //! optimization.
    mutable crf_model_type crf_;

    //! Mapping returned by crf_.conditioned_model_vertex_mapping().
    std::vector
    <typename decomposable<typename crf_factor::output_factor_type>::vertex>
    conditioned_model_vertex_map_;

    //! Temp storage for copies of the CRF factor weights.
    mutable opt_variables crf_tmp_weights;

    //! Count of iterations of parameter learning.
    size_t iteration_;

    //! Uniform distribution over [0, dataset size)
    mutable boost::uniform_int<int> unif_int;

    //! Random number generator.
    mutable boost::mt11213b rng;

    //! For batch optimization methods
    everything_functor* everything_functor_ptr;

    //! For stochastic optimization methods
    stochastic_everything_functor* stochastic_everything_functor_ptr;

    //! For batch optimization methods
    gradient_method_type* gradient_method_ptr;

    //! For stochastic optimization methods
    stochastic_gradient_type* stochastic_gradient_ptr;

    //! Total weight of training data
    double total_train_weight;

    //! Initial training data objective value (log likelihood + regularization).
    double init_train_obj;

    //! Current training data objective value (log likelihood + regularization).
    double train_obj;

    //! Number of calls made to my_objective().
    mutable size_t my_objective_count_;

    //! Number of calls made to my_gradient().
    mutable size_t my_gradient_count_;

    //! Number of calls made to my_stochastic_gradient().
    mutable size_t my_stochastic_gradient_count_;

    //! Number of calls made to my_hessian_diag().
    mutable size_t my_hessian_diag_count_;

    //! Number of calls made to my_everything() without computing the diagonal
    //! of the Hessian.
    mutable size_t my_everything_no_hd_count_;

    //! Number of calls made to my_everything() with computing the diagonal
    //! of the Hessian.
    mutable size_t my_everything_with_hd_count_;

    // Private methods
    // =========================================================================

    //! Initialize learner.
    void init() {
      assert(ds.size() > 0);
      assert(params.valid());
      regularization.regularization = params.regularization;
      if (crf_factor_reg_type::nlambdas != params.lambdas.size()) {
        if (params.lambdas.size() == 1) {
          regularization.lambdas = params.lambdas[0];
        } else {
          throw std::invalid_argument
            (std::string("crf_parameter_learner was given parameters with")
             + " regularization parameters (lambdas) of length "
             + to_string(params.lambdas.size())
             + " but needed lambdas of length "
             + to_string(regularization.lambdas.size()));
        }
      } else {
        regularization.lambdas = params.lambdas;
      }
      rng.seed(params.random_seed);
      unif_int = boost::uniform_int<int>(0, ds.size() - 1);
      for (size_t i(0); i < ds.size(); ++i)
        total_train_weight += ds.weight(i);
      assert(total_train_weight > 0);
    }

    /**
     * Initialize the everything functor pointers and the optimization
     * method pointers the chosen optimization method.
     * Free pointers which are no longer needed.
     */
    void init_pointers() {
      clear_pointers();
      switch(params.opt_method) {
      case real_optimizer_builder::GRADIENT_DESCENT:  // Batch methods first
      case real_optimizer_builder::CONJUGATE_GRADIENT:
      case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
      case real_optimizer_builder::LBFGS:
        everything_functor_ptr =
          new everything_functor(*this, params.no_shared_computation);
        break;
      case real_optimizer_builder::STOCHASTIC_GRADIENT:
        stochastic_everything_functor_ptr =
          new stochastic_everything_functor(*this);
        break;
      default:
        assert(false);
      }

      switch(params.opt_method) {
      case real_optimizer_builder::GRADIENT_DESCENT:
        {
          gradient_descent_parameters ga_params(params.gm_params);
          gradient_method_ptr =
            new gradient_descent_type
            (*everything_functor_ptr, *everything_functor_ptr, crf_.weights(),
             ga_params);
        }
        break;
      case real_optimizer_builder::CONJUGATE_GRADIENT:
        {
          conjugate_gradient_parameters cg_params(params.gm_params);
          gradient_method_ptr =
            new conjugate_gradient_type
            (*everything_functor_ptr, *everything_functor_ptr, crf_.weights(),
             cg_params);
        }
        break;
      case real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC:
        {
          conjugate_gradient_parameters cg_params(params.gm_params);
          gradient_method_ptr =
            new prec_conjugate_gradient_type
            (*everything_functor_ptr, *everything_functor_ptr,
             *everything_functor_ptr, crf_.weights(), cg_params);
        }
        break;
      case real_optimizer_builder::LBFGS:
        {
          lbfgs_parameters lbfgs_params(params.gm_params);
          gradient_method_ptr =
            new lbfgs_type
            (*everything_functor_ptr, *everything_functor_ptr, crf_.weights(),
             lbfgs_params);
        }
        break;
      case real_optimizer_builder::STOCHASTIC_GRADIENT:
        {
          stochastic_gradient_parameters sg_params;
          if (params.init_iterations != 0)
            sg_params.step_multiplier =
              std::exp(std::log(.0001) / params.init_iterations);
          stochastic_gradient_ptr =
            new stochastic_gradient_type
            (*stochastic_everything_functor_ptr, crf_.weights(), sg_params);
        }
        break;
      default:
        assert(false);
      }
    } // init_pointers()

    //! Free pointers with data owned by this class.
    void clear_pointers() {
      if (everything_functor_ptr)
        delete(everything_functor_ptr);
      everything_functor_ptr = NULL;
      if (stochastic_everything_functor_ptr)
        delete(stochastic_everything_functor_ptr);
      stochastic_everything_functor_ptr = NULL;
      if (gradient_method_ptr)
        delete(gradient_method_ptr);
      gradient_method_ptr = NULL;
      if (stochastic_gradient_ptr)
        delete(stochastic_gradient_ptr);
      stochastic_gradient_ptr = NULL;
    }

    /**
     * Finish the initialization.
     * @param init_weights  If true, this sets the initial weights at 0,
     *                      perturbed according to the parameters.
     */
    void init_finish(bool init_weights) {
      if (!crf_.set_log_space(true))
        assert(false);

      if (init_weights) {
        if (params.perturb > 0) {
          throw std::invalid_argument
            (std::string("crf_parameter_learner") +
             " told to init weights, but random initialization has not yet" +
             " been implemented.");
          // NOTE: I need to add a values() function to the OptimizationVector concept to do this.
          /*
          boost::uniform_real<double> uniform(- params.perturb, params.perturb);
          foreach(const crf_graph<>::vertex& v, crf_.factor_vertices()) {
            double w(uniform(rng));
            crf_[v].weight(w);
          }
          */
        } else {
          crf_.weights().zeros();
        }
      }

      ds_it.reset();
      crf_.fix_records(*ds_it);

      ds_it.reset();
      try {
        crf_.condition(*ds_it);
      } catch (normalization_error exc) {
        throw normalization_error
          (std::string("crf_parameter_learner::init_finish()") +
           " could not normalize the CRF given the initial parameter settings" +
           " (Message from normalization attempt: " + exc.what() + ")");
      }
      conditioned_model_vertex_map_ = crf_.conditioned_model_vertex_mapping();

      init_pointers();

      if (gradient_method_ptr) {
        train_obj = gradient_method_ptr->objective();
      } else if (stochastic_gradient_ptr) {
      } else {
        assert(false);
      }
      init_train_obj = train_obj;

      boost::timer timer;
      for (size_t i(0); i < params.init_iterations; ++i) {
        if (!step()) {
          if (params.debug > 0) {
            std::cerr << "crf_parameter_learner::init_finish() terminating"
                      << " after step() returned false on iteration "
                      << iteration_ << ";\n";
            print_stats(std::cerr);
            std::cerr << std::endl;
          }
          if (!params.keep_fixed_records)
            crf_.unfix_records();
          return;
        }
        if (params.init_time_limit != 0 &&
            timer.elapsed() >= params.init_time_limit) {
          if (params.debug > 0) {
            std::cerr << "crf_parameter_learner::init_finish() terminating"
                      << " after exceeding init_time_limit on "
                      << iteration_ << ";\n";
            print_stats(std::cerr);
            std::cerr << std::endl;
          }
          if (!params.keep_fixed_records)
            crf_.unfix_records();
          return;
        }
      }
      if (params.debug > 0) {
        std::cerr << "crf_parameter_learner::init_finish() terminating"
                  << " after init_iterations;\n";
        print_stats(std::cerr);
        std::cerr << std::endl;
      }
      if (!params.keep_fixed_records)
        crf_.unfix_records();
    } // init_finish()

    //! Computes the regularized log likelihood (i.e., optimization objective)
    //! of the training data using CRF factor weights x.
    double my_objective(const opt_variables& x) const {
      ++my_objective_count_;
      double ll(0);
      size_t i(0);
      ds_it.reset();
      crf_tmp_weights = crf_.weights();
      crf_.weights() = x;
      while (ds_it != ds_end) {
        ll -= ds.weight(i) * crf_.log_likelihood(*ds_it);
        ++i;
        ++ds_it;
      }
      foreach(const crf_factor& f, crf_.factors()) {
        ll -= f.regularization_penalty(regularization);
      }
      ll /= total_train_weight;
      crf_.weights() = crf_tmp_weights;
      if (params.debug > 2)
        std::cerr << "crf_parameter_learner::my_objective() called;"
                  << " objective = " << ll << std::endl;
      return ll;
    } // my_objective()

    //! Computes the gradient of the objective at x.
    //! @param gradient  Place in which to store the gradient.
    void my_gradient(opt_variables& gradient, const opt_variables& x) const {
      ++my_gradient_count_;
      assert(gradient.size() == crf_.weights().size());

      if (params.debug > 2)
        std::cerr << "crf_parameter_learner::my_gradient() called."
                  << std::endl;

      gradient = 0;

      size_t i(0);
      ds_it.reset();
      crf_tmp_weights = crf_.weights();
      crf_.weights() = x;
      while (ds_it != ds_end) {
        const decomposable<typename crf_factor::output_factor_type>& Ymodel =
          crf_.condition(*ds_it);
        size_t j(0);
        foreach(const crf_factor& f, crf_.factors()) {
          if (f.fixed_value())
            continue;
          const typename crf_factor::output_factor_type& tmp_marginal
            = Ymodel.marginal(conditioned_model_vertex_map_[j]);
          if (tmp_marginal.arguments().size() == f.output_arguments().size()) {
            f.add_combined_gradient(gradient.factor_weight(j), *ds_it,
                                    tmp_marginal, - ds.weight(i));
          } else {
            typename crf_factor::output_factor_type
              f_marginal(tmp_marginal.marginal(f.output_arguments()));
            f.add_combined_gradient(gradient.factor_weight(j), *ds_it,
                                    f_marginal, - ds.weight(i));
          }
          ++j;
        }
        ++i;
        ++ds_it;
      }

      size_t j(0);
      foreach(const crf_factor& f, crf_.factors()) {
        if (f.fixed_value())
          continue;
        f.add_regularization_gradient(gradient.factor_weight(j),
                                      regularization, -1);
        ++j;
      }
      gradient /= total_train_weight;

      crf_.weights() = crf_tmp_weights;
    } // my_gradient()

    /**
     * Computes the gradient of the objective at x by sampling a single record.
     * @param gradient  Place in which to store the gradient.
     * @return  Objective for the chosen datapoint.
     * @todo Add better support for weighted datasets (using a tree_sampler).
     */
    double my_stochastic_gradient(opt_variables& gradient,
                                  const opt_variables& x) const {
      ++my_stochastic_gradient_count_;
      assert(gradient.size() == crf_.weights().size());

      if (params.debug > 2)
        std::cerr << "crf_parameter_learner::my_stochastic_gradient() called."
                  << std::endl;

      gradient = 0;

      size_t i(unif_int(rng));
      ds_it.reset(i);
      const record_type& r = *ds_it;
      crf_tmp_weights = crf_.weights();
      crf_.weights() = x;
      const decomposable<typename crf_factor::output_factor_type>& Ymodel =
        crf_.condition(r);
      size_t j(0);
      foreach(const crf_factor& f, crf_.factors()) {
        if (f.fixed_value())
          continue;
        const typename crf_factor::output_factor_type& tmp_marginal
          = Ymodel.marginal(conditioned_model_vertex_map_[j]);
        if (tmp_marginal.arguments().size() == f.output_arguments().size()) {
          f.add_combined_gradient(gradient.factor_weight(j), r, tmp_marginal,
                                  -1);
        } else {
          typename crf_factor::output_factor_type
            f_marginal(tmp_marginal.marginal(f.output_arguments()));
          f.add_combined_gradient(gradient.factor_weight(j), r, f_marginal,
                                  -1);
        }
        ++j;
      }
      double neg_ll(- crf_.log_likelihood(r));

      j = 0;
      foreach(const crf_factor& f, crf_.factors()) {
        if (f.fixed_value())
          continue;
        f.add_regularization_gradient(gradient.factor_weight(j),
                                      regularization, -1);
        neg_ll -= f.regularization_penalty(regularization);
        ++j;
      }

      crf_.weights() = crf_tmp_weights;
      return neg_ll;
    } // my_stochastic_gradient()

    //! Computes the diagonal of a Hessian of the function at x.
    //! @param hd  Place in which to store the diagonal.
    void
    my_hessian_diag(opt_variables& hd, const opt_variables& x) const {
      ++my_hessian_diag_count_;
      assert(hd.size() == crf_.weights().size());

      if (params.debug > 2)
        std::cerr << "crf_parameter_learner::my_hessian_diag() called."
                  << std::endl;

      hd = 0;

      size_t i(0);
      ds_it.reset();
      crf_tmp_weights = crf_.weights();
      crf_.weights() = x;
      while (ds_it != ds_end) {
        const decomposable<typename crf_factor::output_factor_type>&
          Ymodel = crf_.condition(ds_it->assignment());
        size_t j(0);
        foreach(const crf_factor& f, crf_.factors()) {
          if (f.fixed_value())
            continue;
          f.add_hessian_diag(hd.factor_weight(j), *ds_it, - ds.weight(i));
          const typename crf_factor::output_factor_type& tmp_marginal
            = Ymodel.marginal(conditioned_model_vertex_map_[j]);
          typename crf_factor::optimization_vector
            tmpoptvec(hd.factor_weight(j).size(), 0.);
          if (tmp_marginal.arguments().size() == f.output_arguments().size()) {
            f.add_expected_hessian_diag(hd.factor_weight(j), *ds_it,
                                        tmp_marginal, ds.weight(i));
            f.add_expected_squared_gradient(hd.factor_weight(j), *ds_it,
                                            tmp_marginal, ds.weight(i));
            f.add_expected_gradient(tmpoptvec, *ds_it, tmp_marginal);
            /*
            f.add_expected_gradient(tmpoptvec, *ds_it, tmp_marginal,
                                    ds.weight(i));
            */
          } else {
            typename crf_factor::output_factor_type
              f_marginal(tmp_marginal.marginal(f.output_arguments()));
            f.add_expected_hessian_diag(hd.factor_weight(j), *ds_it, f_marginal,
                                        ds.weight(i));
            f.add_expected_squared_gradient(hd.factor_weight(j), *ds_it,
                                            f_marginal, ds.weight(i));
            f.add_expected_gradient(tmpoptvec, *ds_it, f_marginal);
            /*
            f.add_expected_gradient(tmpoptvec, *ds_it, f_marginal,
                                    ds.weight(i));
            */
          }
          tmpoptvec.elem_mult(tmpoptvec);
//          hd.factor_weight(j) -= tmpoptvec;
          hd.factor_weight(j) -= (ds.weight(i) == 1 ?
                                  tmpoptvec :
                                  tmpoptvec * ds.weight(i));
          ++j;
        }
        ++i;
        ++ds_it;
      }

      size_t j(0);
      foreach(const crf_factor& f, crf_.factors()) {
        if (f.fixed_value())
          continue;
        f.add_regularization_hessian_diag(hd.factor_weight(j),
                                          regularization, -1);
        ++j;
      }
      hd /= total_train_weight;

      crf_.weights() = crf_tmp_weights;
    } // my_hessian_diag()

    /**
     * Computes the objective, gradient and (if needed) the diagonal of the
     * Hessian at x.
     * @param codes  Indicates which things to compute:
     *                - 0: all 3
     *                - 1: objective and gradient only
     */
    void
    my_everything(double& obj, opt_variables& gradient, opt_variables& hd,
                  const opt_variables& x, size_t codes) const {
      assert(gradient.size() == crf_.weights().size());
      if (codes == 0) {
        assert(params.opt_method ==
               real_optimizer_builder::CONJUGATE_GRADIENT_DIAG_PREC);
        assert(hd.size() == crf_.weights().size());
        ++my_everything_with_hd_count_;
      } else {
        ++my_everything_no_hd_count_;
      }
      if (params.debug > 2)
        std::cerr << "crf_parameter_learner::my_everything() called."
                  << std::endl;

      obj = 0.;
      gradient = 0;
      if (codes == 0)
        hd = 0;

      size_t i(0);
      ds_it.reset();
      crf_tmp_weights = crf_.weights();
      crf_.weights() = x;
      while (ds_it != ds_end) {
        const record_type& rec = *ds_it;
        const decomposable<typename crf_factor::output_factor_type>&
          Ymodel = crf_.condition(rec);
        obj -= ds.weight(i) * Ymodel.log_likelihood(rec);
        size_t j(0);
        foreach(const crf_factor& f, crf_.factors()) {
          if (f.fixed_value())
            continue;
          const typename crf_factor::output_factor_type& tmp_j_marginal =
            Ymodel.marginal(conditioned_model_vertex_map_[j]);
          typename crf_factor::output_factor_type* tmp_marginal_ptr = NULL;
          if (tmp_j_marginal.size() != f.output_arguments().size()) {
            tmp_marginal_ptr = new typename crf_factor::output_factor_type
              (tmp_j_marginal.marginal(f.output_arguments()));
          }
          const typename crf_factor::output_factor_type& tmp_marginal =
            (tmp_j_marginal.size() == f.output_arguments().size() ?
             tmp_j_marginal : *tmp_marginal_ptr);

          if (codes == 1) {
            f.add_combined_gradient(gradient.factor_weight(j), *ds_it,
                                    tmp_marginal, - ds.weight(i));
          } else if (codes == 0) {
            f.add_gradient(gradient.factor_weight(j), *ds_it,
                           - ds.weight(i));
            f.add_hessian_diag(hd.factor_weight(j), *ds_it, -ds.weight(i));
            f.add_expected_hessian_diag(hd.factor_weight(j), *ds_it,
                                        tmp_marginal, ds.weight(i));
            f.add_expected_squared_gradient(hd.factor_weight(j), *ds_it,
                                            tmp_marginal,
                                            ds.weight(i));
            typename crf_factor::optimization_vector
              tmpoptvec(hd.factor_weight(j).size(), 0.);
            /*
            f.add_expected_gradient(tmpoptvec, *ds_it, tmp_marginal,
                                    ds.weight(i));
            gradient.factor_weight(j) += tmpoptvec;
            tmpoptvec.elem_mult(tmpoptvec);
            hd.factor_weight(j) -= tmpoptvec;
            */
            f.add_expected_gradient(tmpoptvec, *ds_it, tmp_marginal);
            gradient.factor_weight(j) += (ds.weight(i) == 1 ?
                                          tmpoptvec :
                                          tmpoptvec * ds.weight(i));
            tmpoptvec.elem_mult(tmpoptvec);
            hd.factor_weight(j) -= (ds.weight(i) == 1 ?
                                    tmpoptvec :
                                    tmpoptvec * ds.weight(i));
          } else {
            assert(false);
          }
          if (tmp_marginal_ptr) {
            delete(tmp_marginal_ptr);
            tmp_marginal_ptr = NULL;
          }
          ++j;
        }
        ++i;
        ++ds_it;
      }

      size_t j(0);
      foreach(const crf_factor& f, crf_.factors()) {
        obj -= f.regularization_penalty(regularization);
        if (!f.fixed_value()) {
          f.add_regularization_gradient(gradient.factor_weight(j),
                                        regularization, -1);
          if (codes == 0) {
            f.add_regularization_hessian_diag(hd.factor_weight(j),
                                              regularization, -1);
          }
          ++j;
        }
      }

      obj /= total_train_weight;
      gradient /= total_train_weight;
      if (codes == 0)
        hd /= total_train_weight;

      crf_.weights() = crf_tmp_weights;
      if (params.debug > 2)
        std::cerr << "crf_parameter_learner::my_everything() computed"
                  << " objective = " << obj << std::endl;
    } // my_everything()

    // Constructors and destructors
    // =========================================================================
  public:

    /*
     * Initializes a CRF model learner using the given graph structure.
     */
    crf_parameter_learner(const typename crf_model_type::crf_graph_type& graph,
                          const dataset<la_type>& ds,
                          const parameters& params = parameters())
      : params(params),
        ds(ds), ds_it(ds.begin()), ds_end(ds.end()),
        crf_(graph), crf_tmp_weights(crf_.weights()),
        iteration_(0),
        everything_functor_ptr(NULL), stochastic_everything_functor_ptr(NULL),
        gradient_method_ptr(NULL), stochastic_gradient_ptr(NULL),
        total_train_weight(0),
        init_train_obj(std::numeric_limits<double>::max()),
        train_obj(std::numeric_limits<double>::max()),
        my_objective_count_(0), my_gradient_count_(0),
        my_stochastic_gradient_count_(0),
        my_hessian_diag_count_(0), my_everything_no_hd_count_(0),
        my_everything_with_hd_count_(0) {
      init();
      init_finish(true);
    }

    /**
     * Initializes a CRF model learner using the given model,
     * which specifies features.
     * @param keep_weights  If false, the model weights are re-initialized.
     */
    crf_parameter_learner(const crf_model_type& model,
                          const dataset<la_type>& ds,
                          bool keep_weights,
                          const parameters& params = parameters())
      : params(params),
        ds(ds), ds_it(ds.begin()), ds_end(ds.end()),
        crf_(model), crf_tmp_weights(crf_.weights()),
        iteration_(0), everything_functor_ptr(NULL),
        stochastic_everything_functor_ptr(NULL),
        gradient_method_ptr(NULL), stochastic_gradient_ptr(NULL),
        total_train_weight(0),
        init_train_obj(std::numeric_limits<double>::max()),
        train_obj(std::numeric_limits<double>::max()),
        my_objective_count_(0), my_gradient_count_(0),
        my_stochastic_gradient_count_(0),
        my_hessian_diag_count_(0), my_everything_no_hd_count_(0),
        my_everything_with_hd_count_(0) {
      init();
      if (keep_weights)
        init_finish(false);
      else
        init_finish(true);
    }

    ~crf_parameter_learner() {
      clear_pointers();
    }

    //! Return the current model.
    const crf_model_type& current_model() const {
      return crf_;
    }

    /**
     * Do one step of parameter learning.
     * @return  false if the step was unsuccessful (e.g., if the parameters
     *          have converged)
     */
    bool step() {
      if (gradient_method_ptr) {
        double prev_train_obj(train_obj);
        if (!gradient_method_ptr->step())
          return false;
        train_obj = gradient_method_ptr->objective();
        if (params.debug > 1) {
          if (train_obj > prev_train_obj)
            std::cerr << "crf_parameter_learner took a step which "
                      << "increased the objective from " << prev_train_obj
                      << " to " << train_obj << std::endl;
          std::cerr << "change in objective = "
                    << (train_obj - prev_train_obj) << std::endl;
        }
        // Check for convergence
        if (fabs(train_obj - prev_train_obj)
            < params.gm_params.convergence_zero) {
          if (params.debug > 1)
            std::cerr << "crf_parameter_learner converged:"
                      << " training objective changed from "
                      << prev_train_obj << " to " << train_obj
                      << "; exiting on iteration " << iteration() << "."
                      << std::endl;
          return false;
        }
      } else if (stochastic_gradient_ptr) {
        if (!stochastic_gradient_ptr->step())
          return false;
      } else {
        assert(false);
      }
      ++iteration_;
      return true;
    } // end of step()

    //! Iteration number (i.e., how many iterations have been completed).
    //! This is also the number of times the gradient (and preconditioner)
    //! have been computed.
    size_t iteration() const {
      return iteration_;
    }

    //! Return the average number of objective function calls per iteration,
    //! or -1 if no iterations have completed, if the underlying optimization
    //! routine supports this.  If not, returns -2.
    double objective_calls_per_iteration() const {
      if (!gradient_method_ptr)
        return -2;
      return gradient_method_ptr->objective_calls_per_iteration();
    }

    //! Number of calls made to my_objective().
    size_t my_objective_count() const {
      return my_objective_count_;
    }

    //! Number of calls made to my_gradient().
    size_t my_gradient_count() const {
      return my_gradient_count_;
    }

    //! Number of calls made to my_stochastic_gradient().
    size_t my_stochastic_gradient_count() const {
      return my_stochastic_gradient_count_;
    }

    //! Number of calls made to my_hessian_diag().
    size_t my_hessian_diag_count() const {
      return my_hessian_diag_count_;
    }

    //! Number of calls made to my_everything() without computing the diagonal
    //! of the Hessian.
    size_t my_everything_no_hd_count() const {
      return my_everything_no_hd_count_;
    }

    //! Number of calls made to my_everything() with computing the diagonal
    //! of the Hessian.
    size_t my_everything_with_hd_count() const {
      return my_everything_with_hd_count_;
    }

    //! Print debugging info about calls to objective, gradient, etc.,
    //! as well as objective info (if available).
    void print_stats(std::ostream& out) const {
      if (gradient_method_ptr)
        out << " Initial objective: " << init_train_obj << "\n"
            << " Current objective: " << gradient_method_ptr->objective()
            << "\n";
      out << " Method calls:\n"
          << "\tmy_objective:             " << my_objective_count() << "\n"
          << "\tmy_gradient:              " << my_gradient_count() << "\n"
          << "\tmy_stochastic_gradient:   " << my_stochastic_gradient_count()
          << "\n"
          << "\tmy_hessian_diag:          " << my_hessian_diag_count() << "\n"
          << "\tmy_everything without hd: " << my_everything_no_hd_count()
          << "\n"
          << "\tmy_everything with hd:    " << my_everything_with_hd_count()
          << "\n";
    }

    //! Returns the parameters (which may be modified by cross-validation
    //! for parameter tuning).
    const parameters& get_params() const {
      return params;
    }

    /**
     * Choose regularization parameters via n-fold cross validation.
     *
     * @param cv_params   Parameters specifying how to do cross validation.
     * @param model       CRF model on which to do parameter learning.
     * @param keep_weights  If true, keep weights in model; if false, set to 0.
     * @param ds          Training data.
     * @param params      Parameters for this class.
     * @param score_type  0: log likelihood, 1: per-label accuracy,
     *                    2: all-or-nothing label accuracy,
     *                    3: mean squared error
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  chosen regularization parameters
     */
    static
    vec
    choose_lambda(const crossval_parameters& cv_params,
                  const crf_model_type& model, bool keep_weights,
                  const dataset<la_type>& ds, const parameters& params,
                  size_t score_type, unsigned random_seed) {
      assert(score_type == 0); // others not yet implemented
      crf_validation_functor<F> crf_val_func(model, params, keep_weights);
      validation_framework<la_type>
        val_frame(ds, cv_params, crf_val_func, random_seed);
      return val_frame.best_lambdas();
    }

    /**
     * Choose regularization parameters via n-fold cross validation.
     *
     * @param cv_params   Parameters specifying how to do cross validation.
     * @param structure   CRF structure.
     * @param ds          Training data.
     * @param params      Parameters for this class.
     * @param score_type  0: log likelihood, 1: per-label accuracy,
     *                    2: all-or-nothing label accuracy,
     *                    3: mean squared error
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  chosen regularization parameters
     *
     * WARNING: This version does not work with templated factors.
     *          Use the above choose_lambda instead.
     */
    static
    vec
    choose_lambda(const crossval_parameters& cv_params,
                  const typename crf_model_type::crf_graph_type& structure,
                  const dataset<la_type>& ds, const parameters& params,
                  size_t score_type, unsigned random_seed) {
      assert(score_type == 0); // others not yet implemented
      crf_validation_functor<F> crf_val_func(structure, params);
      validation_framework<la_type>
        val_frame(ds, cv_params, crf_val_func, random_seed);
      return val_frame.best_lambdas();
    }

  }; // crf_parameter_learner

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_CRF_PARAMETER_LEARNER_HPP
