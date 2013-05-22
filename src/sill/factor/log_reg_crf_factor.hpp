#ifndef SILL_LOG_REG_CRF_FACTOR_HPP
#define SILL_LOG_REG_CRF_FACTOR_HPP

#include <sill/base/universe.hpp>
#include <sill/factor/learnable_crf_factor.hpp>
#include <sill/learning/discriminative/multiclass2multilabel.hpp>
#include <sill/learning/discriminative/multiclass_logistic_regression.hpp>
#include <sill/learning/validation/crossval_parameters.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * CRF factor based on multilabel logistic regression.
   * This supports both finite and vector X.
   *
   * This satisfies the LearnableCRFfactor concept.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   *
   * \ingroup factor
   * @author Joseph Bradley
   */
  template <typename LA = dense_linear_algebra<> >
  class log_reg_crf_factor
    : public learnable_crf_factor<variable, table_factor,
                                  typename multiclass_logistic_regression<LA>::opt_variables,
                                  1, LA> {

    // Public types
    // =========================================================================
  public:

    typedef LA la_type;

    //! Base class
    typedef
    learnable_crf_factor
    <variable, table_factor,
     typename multiclass_logistic_regression<la_type>::opt_variables,
     1, la_type>
    base;

    // Import types from base
    typedef typename base::input_variable_type   input_variable_type;
    typedef typename base::input_domain_type     input_domain_type;
    typedef typename base::input_assignment_type input_assignment_type;
    typedef typename base::input_var_vector_type input_var_vector_type;
    typedef typename base::input_var_map_type    input_var_map_type;
    typedef typename base::input_record_type     input_record_type;
    typedef typename base::output_variable_type   output_variable_type;
    typedef typename base::output_domain_type     output_domain_type;
    typedef typename base::output_assignment_type output_assignment_type;
    typedef typename base::output_var_vector_type output_var_vector_type;
    typedef typename base::output_var_map_type    output_var_map_type;
    typedef typename base::output_record_type     output_record_type;
    typedef typename base::variable_type   variable_type;
    typedef typename base::domain_type     domain_type;
    typedef typename base::assignment_type assignment_type;
    typedef typename base::var_vector_type var_vector_type;
    typedef typename base::var_map_type    var_map_type;
    typedef typename base::record_type     record_type;
    typedef typename base::result_type          result_type;
    typedef typename base::output_factor_type   output_factor_type;
    typedef typename base::optimization_vector  optimization_vector;
    typedef typename base::regularization_type  regularization_type;

    //! Parameters used for learning feature values from data.
    struct parameters {

      multiclass_logistic_regression_parameters mlr_params;

      /**
       * Regularization parameters used for learn_crf_factor;
       * these override mlr_params.
       *  (default = regularization_type defaults)
       */
      regularization_type reg;

      /**
       * Amount of smoothing. (>= 0)
       * E.g., 1 = pretend we saw 1 extra training example with each possible
       * assignment, where we think of vector input variables as being binary.
       * (This approximates the notion of smoothing for table_crf_factor.)
       *  (default = 1)
       */
      double smoothing;

      universe& u;

      //! Use the default multiclass_logistic_regression parameters.
      explicit parameters(universe& u) : smoothing(1), u(u) {
      }

      //! Use the given multiclass_logistic_regression parameters.
      parameters(const multiclass_logistic_regression_parameters& mlr_params,
                 universe& u) : mlr_params(mlr_params), smoothing(1), u(u) {
      }

      //! Assignment operator.
      //! (Necessary because of non-static reference member.  This assumes
      //! both parameters structs were initialized with the same universe.)
      parameters& operator=(const parameters& params) {
        mlr_params = params.mlr_params;
        reg = params.reg;
        smoothing = params.smoothing;
        return *this;
      }

      bool valid() const {
        if (smoothing < 0)
          return false;
        return true;
        /*
        if (!mlr_params.valid())
          return false;
        */
      }

    }; // struct parameters

    // Protected data
    // =========================================================================
  protected:

    //! Multilabel logistic regressor
    boost::shared_ptr<multiclass2multilabel<la_type> > mlr_ptr;

    //! Amount of smoothing to add to probabilities produced by the
    //! logistic regressor; note this is different than 'smoothing' in the
    //! parameters.  It is calculated at training time.
    double smoothing;

    //! Temporary to avoid reallocation; used for computing gradients.
    mutable record_type tmp_record;

    //! Temporary used to avoid reallocation for conditioning.
    mutable table_factor conditioned_f;

    // Public methods: Constructors, getters, helpers
    // =========================================================================
  public:
    //! Default constructor.
    log_reg_crf_factor()
      : base() { }

    /**
     * Constructor which initializes the weights to 0.
     * @param Y    Y variables
     * @param X    X variables
     */
    log_reg_crf_factor(const output_domain_type& Y_,
                       const input_domain_type& X_)
      : base(Y_, copy_ptr<domain>(new domain(X_.begin(), X_.end()))),
        conditioned_f(Y_, 0.) {
      assert(false); // I NEED TO INITIALIZE mlr_ptr.
    }

    /**
     * Constructor.
     * @param mlr_ptr      Pointer to a multiclass2multilabel instance
     *                     which uses a multiclass_logistic_regression
     *                     instance as its base classifier.
     * @param Y_           Y variables
     * @param X_ptr_       X variables
     */
    log_reg_crf_factor
    (boost::shared_ptr<multiclass2multilabel<la_type> > mlr_ptr,
     double smoothing, const finite_domain& Y_, copy_ptr<domain> X_ptr_)
      : base(Y_, X_ptr_), mlr_ptr(mlr_ptr), smoothing(smoothing), 
        conditioned_f(Y_, 0.) {
      assert(mlr_ptr.get() != NULL);
      mlr_ptr->prepare_record_for_base(tmp_record);
    }

    void print(std::ostream& out) const;

    //! Returns the dataset structure for records used by this factor.
    const datasource_info_type& datasource_info() const {
      assert(mlr_ptr);
      return mlr_ptr->datasource_info();
    }

    /**
     * This method may not be used with log_reg_crf_factor.
     */
    void relabel_outputs_inputs(const output_domain_type& new_Y,
                                const input_domain_type& new_X) {
      assert(false);
    }

    // Public methods: Probabilistic queries
    // =========================================================================

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param a  This must assign values to all X in this factor
     *           (but may assign values to any other variables as well).
     * @return  table factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const table_factor& condition(const input_assignment_type& a) const;

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param r Record with values for X in this factor
     *          (which may have values for any other variables as well).
     * @return  table factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const table_factor& condition(const input_record_type& r) const;

    // Public: Learning-related methods from crf_factor interface
    // =========================================================================

    //! @return  true iff the data is stored in log-space
    bool log_space() const {
      return true;
    }

    //! Tries to change this factor's internal representation to log-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already log-space
    bool convert_to_log_space() {
      return true;
    }

    //! Tries to change this factor's internal representation to real-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already real-space
    bool convert_to_real_space() {
      return false;
    }

    //! This has not been implemented for this class.
    void fix_records(const record_type& r) {
    }

    //! This has not been implemented for this class.
    void unfix_records() {
    }

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    const optimization_vector& weights() const;

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    optimization_vector& weights();

    // Public: Learning methods from learnable_crf_factor interface
    // =========================================================================

    //! Adds the gradient of the log of this factor w.r.t. the weights,
    //! evaluated at the given datapoint with the current weights.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param w      Weight by which to multiply the added values.
    void add_gradient(optimization_vector& grad,
                      const record_type& r, double w) const;

    /**
     * Adds the expectation of the gradient of the log of this factor
     * w.r.t. the weights, evaluated with the current weights and at the
     * given datapoint for the X values.  The expectation is over the Y
     * values and w.r.t. the given factor's distribution.
     * @param grad   Pre-allocated vector to which to add the gradient.
     * @param r      Datapoint.
     * @param fy     Distribution over (at least) the Y variables in this
     *               factor.
     * @param w      Weight by which to multiply the added values.
     */
    void
    add_expected_gradient(optimization_vector& grad,
                          const input_record_type& r, const table_factor& fy,
                          double w = 1) const;

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void
    add_combined_gradient(optimization_vector& grad, const record_type& r,
                          const output_factor_type& fy, double w = 1) const;

    /**
     * Adds the diagonal of the Hessian of the log of this factor w.r.t. the
     * weights, evaluated at the given datapoint with the current weights.
     * @param hessian Pre-allocated vector to which to add the hessian.
     * @param r       Datapoint.
     * @param w       Weight by which to multiply the added values.
     */
    void
    add_hessian_diag(optimization_vector& hessian, const record_type& r,
                     double w) const;

    /**
     * Adds the expectation of the diagonal of the Hessian of the log of this
     * factor w.r.t. the weights, evaluated with the current weights and at the
     * given datapoint for the X values.  The expectation is over the Y
     * values and w.r.t. the given factor's distribution.
     * @param hessian Pre-allocated vector to which to add the Hessian.
     * @param r       Datapoint.
     * @param fy      Distribution over (at least) the Y variables in this
     *                factor.
     * @param w       Weight by which to multiply the added values.
     */
    void
    add_expected_hessian_diag(optimization_vector& hessian,
                              const input_record_type& r,
                              const table_factor& fy, double w) const;

    /**
     * Adds the expectation of the element-wise square of the gradient of the
     * log of this factor w.r.t. the weights, evaluated with the current
     * weights and at the given datapoint for the X values. The expectation is
     * over the Y values and w.r.t. the given factor's distribution.
     * @param sqrgrad Pre-allocated vector to which to add the squared gradient.
     * @param r       Datapoint.
     * @param fy      Distribution over (at least) the Y variables in this
     *                factor.
     * @param w       Weight by which to multiply the added values.
     */
    void
    add_expected_squared_gradient(optimization_vector& sqrgrad,
                                  const input_record_type& r,
                                  const table_factor& fy, double w) const;

    /**
     * Returns the regularization penalty for the current weights and
     * the given regularization parameters.
     * This is:  - .5 * lambda * inner_product(weights, weights)
     */
    double regularization_penalty(const regularization_type& reg) const;

    /**
     * Adds the gradient of the regularization term for the current weights
     * and the given regularization parameters.
     * This is:  - lambda * weights
     */
    void add_regularization_gradient(optimization_vector& grad,
                                     const regularization_type& reg,
                                     double w) const;

    /**
     * Adds the diagonal of the Hessian of the regularization term for the
     * current weights and the given regularization parameters.
     */
    void add_regularization_hessian_diag(optimization_vector& hd,
                                         const regularization_type& reg,
                                         double w) const;

  };  // class log_reg_crf_factor

  //============================================================================
  // Implementations of methods in log_reg_crf_factor
  //============================================================================


  // Public methods: Constructors, getters, helpers
  // =========================================================================

  template <typename LA>
  void log_reg_crf_factor<LA>::print(std::ostream& out) const {
    base::print(out);
    if (mlr_ptr)
      out << *mlr_ptr;
  }

  // Public methods: Probabilistic queries
  // =========================================================================

  template <typename LA>
  const table_factor&
  log_reg_crf_factor<LA>::condition(const input_assignment_type& a) const {
    conditioned_f = mlr_ptr->probabilities(a);
    conditioned_f += smoothing;
    conditioned_f.normalize();
    return conditioned_f;
    // TO DO: MAKE THE ABOVE MORE EFFICIENT IF NECESSARY (I.E., MAKE USE OF
    //        THE PRE-ALLOCATED conditioned_f.
  }

  template <typename LA>
  const table_factor&
  log_reg_crf_factor<LA>::condition(const input_record_type& r) const {
    conditioned_f = mlr_ptr->probabilities(r);
    conditioned_f += smoothing;
    conditioned_f.normalize();
    return conditioned_f;
    // TO DO: MAKE THE ABOVE MORE EFFICIENT IF NECESSARY (I.E., MAKE USE OF
    //        THE PRE-ALLOCATED conditioned_f.
  }

  // Public: Learning-related methods from crf_factor interface
  // =========================================================================

  template <typename LA>
  const typename log_reg_crf_factor<LA>::optimization_vector&
  log_reg_crf_factor<LA>::weights() const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<la_type> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<la_type>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<la_type>*>(base_ptr.get());
    return base_mlr_ptr->weights();
  }

  template <typename LA>
  typename log_reg_crf_factor<LA>::optimization_vector&
  log_reg_crf_factor<LA>::weights() {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<la_type> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    multiclass_logistic_regression<la_type>* base_mlr_ptr
      = dynamic_cast<multiclass_logistic_regression<la_type>*>(base_ptr.get());
    return base_mlr_ptr->weights();
  }

  // Public: Learning methods from learnable_crf_factor interface
  // =========================================================================

  template <typename LA>
  void
  log_reg_crf_factor<LA>::add_gradient
  (optimization_vector& grad, const record_type& r,
   double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<la_type> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<la_type>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<la_type>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_gradient(grad, tmp_record, w);
  }

  template <typename LA>
  void
  log_reg_crf_factor<LA>::add_expected_gradient
  (optimization_vector& grad,
   const input_record_type& r, const table_factor& fy,
   double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<la_type> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<la_type>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<la_type>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_expected_gradient(grad, tmp_record, fy, w);
  }

  template <typename LA>
  void
  log_reg_crf_factor<LA>::add_combined_gradient
  (optimization_vector& grad, const record_type& r,
   const output_factor_type& fy, double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<la_type> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<la_type>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<la_type>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_combined_gradient(grad, tmp_record, fy, w);
  }

  template <typename LA>
  void
  log_reg_crf_factor<LA>::add_hessian_diag
  (optimization_vector& hessian, const record_type& r, double w) const {
    return; // This is 0.
  }

  template <typename LA>
  void
  log_reg_crf_factor<LA>::add_expected_hessian_diag
  (optimization_vector& hessian, const input_record_type& r,
   const table_factor& fy, double w) const {
    return; // This is 0.
  }

  template <typename LA>
  void
  log_reg_crf_factor<LA>::add_expected_squared_gradient
  (optimization_vector& sqrgrad, const input_record_type& r,
   const table_factor& fy, double w) const {
    assert(mlr_ptr);
    boost::shared_ptr<multiclass_classifier<la_type> >
      base_ptr(mlr_ptr->get_base_learner_ptr());
    const multiclass_logistic_regression<la_type>* base_mlr_ptr
      = dynamic_cast<const multiclass_logistic_regression<la_type>*>(base_ptr.get());
    mlr_ptr->convert_record_for_base(r, tmp_record);
    base_mlr_ptr->add_expected_squared_gradient(sqrgrad, tmp_record, fy, w);
  }

  template <typename LA>
  double log_reg_crf_factor<LA>::regularization_penalty
  (const regularization_type& reg) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return 0.;
    case 2:
      if (reg.lambdas[0] == 0) {
        return 0.;
      } else {
        const optimization_vector& tmpov = weights();
        return (-.5 * reg.lambdas[0] * tmpov.dot(tmpov));
      }
    default:
      throw std::invalid_argument("log_reg_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  template <typename LA>
  void log_reg_crf_factor<LA>::add_regularization_gradient
  (optimization_vector& grad, const regularization_type& reg, double w) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return;
    case 2:
      {
        const optimization_vector& tmpov = weights();
        if (reg.lambdas[0] != 0)
          grad -= tmpov * w * reg.lambdas[0];
      }
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

  template <typename LA>
  void log_reg_crf_factor<LA>::add_regularization_hessian_diag
  (optimization_vector& hd, const regularization_type& reg, double w) const {
    assert(reg.lambdas.size() == reg.nlambdas);
    switch(reg.regularization) {
    case 0:
      return;
    case 2:
      if (reg.lambdas[0] != 0)
        hd -= w * reg.lambdas[0];
      return;
    default:
      throw std::invalid_argument("table_crf_factor::regularization_penalty() given bad regularization argument.");
    }
  }

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LOG_REG_CRF_FACTOR_HPP
