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
   * \ingroup factor
   * @author Joseph Bradley
   */
  class log_reg_crf_factor
    : public learnable_crf_factor<variable, table_factor,
                                  multiclass_logistic_regression<>::opt_variables,
                                  1> {

    // Public types
    // =========================================================================
  public:

    //! Base class
    typedef
    learnable_crf_factor<variable, table_factor,
                         multiclass_logistic_regression<>::opt_variables, 1>
    base;

    typedef base::la_type la_type;
    typedef record<la_type> record_type;

    //! Parameters used for learning feature values from data.
    struct parameters {

      multiclass_logistic_regression_parameters mlr_params;

      /**
       * Regularization parameters used for learn_crf_factor();
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
    boost::shared_ptr<multiclass2multilabel> mlr_ptr;

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
                       const input_domain_type& X_);

    /**
     * Constructor.
     * @param mlr_ptr      Pointer to a multiclass2multilabel instance
     *                     which uses a multiclass_logistic_regression
     *                     instance as its base classifier.
     * @param Y_           Y variables
     * @param X_ptr_       X variables
     */
    log_reg_crf_factor(boost::shared_ptr<multiclass2multilabel> mlr_ptr,
                       double smoothing, const finite_domain& Y_,
                       copy_ptr<domain> X_ptr_);

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
    const table_factor& condition(const assignment& a) const;

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param r Record with values for X in this factor
     *          (which may have values for any other variables as well).
     * @return  table factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const table_factor& condition(const record_type& r) const;

    /**
     * Returns the empirical expectation of the log of this factor.
     * In particular, if this factor represents P(A|B), then
     * this returns the expected log likelihood of the distribution P(A | B).
     * This uses real-space; i.e., the log of this factor is in log-space.
     */
//    double log_expected_value(const dataset& ds) const;

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
    const multiclass_logistic_regression<>::opt_variables& weights() const;

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    multiclass_logistic_regression<>::opt_variables& weights();

    // Public: Learning methods from learnable_crf_factor interface
    // =========================================================================

    //! Adds the gradient of the log of this factor w.r.t. the weights,
    //! evaluated at the given datapoint with the current weights.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param w      Weight by which to multiply the added values.
    void add_gradient(multiclass_logistic_regression<>::opt_variables& grad,
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
    add_expected_gradient(multiclass_logistic_regression<>::opt_variables& grad,
                          const record_type& r, const table_factor& fy,
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
    add_expected_hessian_diag(optimization_vector& hessian, const record_type& r,
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
    add_expected_squared_gradient(optimization_vector& sqrgrad, const record_type& r,
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

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LOG_REG_CRF_FACTOR_HPP
