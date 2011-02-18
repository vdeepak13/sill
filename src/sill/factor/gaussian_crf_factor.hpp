#ifndef SILL_GAUSSIAN_CRF_FACTOR_HPP
#define SILL_GAUSSIAN_CRF_FACTOR_HPP

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/learnable_crf_factor.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/validation/crossval_parameters.hpp>
#include <sill/learning/discriminative/linear_regression.hpp>
#include <sill/math/linear_algebra_errors.hpp>
#include <sill/optimization/gaussian_opt_vector.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declarations
  class canonical_gaussian;
  class moment_gaussian;

  /**
   * CRF factor which produces Gaussian factors when conditioned.
   * This supports vector Y,X only.
   *
   * This represents a function f(Y,X) = exp(-(1/2)(AY - (b + CX))^2),
   * which may be converted into a Gaussian factor for P(Y | X = x).
   * (Note that the covariance matrix Sigma = (A'A)^-1.)
   *
   * This satisfies the LearnableCRFfactor concept.
   *
   * Regularization info:
   *  - Method (regularization_type::regularization)
   *     - 0: none
   *     - 2: (1/2) * [lambdas[0] * (L2norm(b)^2 + FrobeniusNorm(C)^2)
   *                   + lambdas[1] * FrobeniusNorm(A)^2]
   *          Note: FrobeniusNorm(A)^2 = tr(A'A)
   *          (Part of an inverse Wishart prior on A)
   *         (default)
   *     - 3: (lambdas[0]/2) * (L2norm(b)^2 + FrobeniusNorm(C)^2)
   *          + ((lambdas[1] + 2*d)/2) logdet((A'A)^-1)
   *          (Part of an inverse Wishart prior on A)
   *     - 4: (lambdas[0]/2) * (L2norm(b)^2 + FrobeniusNorm(C)^2)
   *          + ((lambdas[1] + 2*d)/2) logdet((A'A)^-1)
   *          + (lambdas[1]/2) * tr(A'A)
   *          (Particular inverse Wishart prior on A)
   *     - 5: (1/2) * [lambdas[0] * (L2norm(b)^2 + FrobeniusNorm(C)^2)
   *                   + lambdas[1] * tr((A'A)^-1)]
   *          (Part of a Wishart prior on A)
   *     - 6: (1/2) * [lambdas[0] * (L2norm(b)^2 + FrobeniusNorm(C)^2)
   *                   + lambdas[1] [ -logdet((A'A)^-1) + tr((A'A)^-1) ] ]
   *          (Particular Wishart prior on A)
   *  - Lambdas (regularization_type::lambdas)
   *     - These are applied slightly differently for closed-form learning of a
   *       conditional Gaussian vs. regularization using crf_parameter_learner.
   *     - For closed-form learning, these are interpreted as follows:
   *        - lambdas[0]: (lambda_bC)
   *            Use a Normal prior (i.e., L2 regularization) on the mean
   *            and coefficients (in the moment Gaussian representation).
   *            (Essentially, use a Normal(mu; 0, (1/lambdas[0]) * I) prior.)
   *        - lambdas[1]: (lambda_cov)
   *            If > 0, use a Wishart prior on the covariance:
   *             P(sigma | mu) = W(Sigma | mu; Lambda, alpha)
   *               where Lambda = lambdas[1] * I, alpha = d and
   *               where the mu is determined using the Normal prior above.
   *     - For crf_parameter_learner, these are interpreted according to the
   *       regularization parameter above.
   *     - Note that the lambdas are proportional to numbers of pseudoexamples.
   *
   * \ingroup factor
   * @author Joseph Bradley
   */
  class gaussian_crf_factor
    : public learnable_crf_factor<vector_variable, canonical_gaussian,
                                  gaussian_opt_vector, 2> {

    // Public types
    // =========================================================================
  public:

    //! Base class
    typedef learnable_crf_factor<vector_variable, canonical_gaussian,
                                 gaussian_opt_vector, 2> base;

    //! Parameters used for learn_crf_factor().
    struct parameters {

      /**
       * Regularization parameters used for learn_crf_factor().
       *  (defaults = regularization_type defaults)
       */
      regularization_type reg;

      /**
       * This permits learn_crf_factor() to increase the regularization
       * parameters as necessary to deal with numerical issues.
       * The parameters may be increased up to this value (>= 0).
       *  (default = 0)
       */
      double max_lambda_cov;

      //! Number of increments this tries between lambda_cov and max_lambda_cov,
      //! excluding lambda_cov and including max_lambda_cov.
      //! Must be > 0.
      //!  (default = 3)
      size_t lambda_cov_increments;

      /**
       * Cross validation score type:
       *  - 0: Mean of CV score.
       *     (default)
       *  - 1: Mean of CV score - standard error of CV score.
       */
      size_t cv_score_type;

      /**
       * Print debugging info.
       *  - 0: none (default)
       *  - 1: some
       *  - higher values: default to highest debugging mode
       */
      size_t debug;

      parameters()
        : max_lambda_cov(0), lambda_cov_increments(3), cv_score_type(0),
          debug(0) { }

      bool valid() const {
        if (max_lambda_cov < 0)
          return false;
        if (lambda_cov_increments == 0)
          return false;
        if (cv_score_type > 1)
          return false;
        return true;
      }

    }; // struct parameters

    // Public methods: Constructors, getters, helpers
    // =========================================================================

    //! Default constructor.
    gaussian_crf_factor();

    /**
     * Constructor which initializes the weights to 0.
     * @param Y    Y variables
     * @param X    X variables
     */
    gaussian_crf_factor(const forward_range<vector_variable*>& Y_,
                        const forward_range<vector_variable*>& X_);

    /**
     * Constructor which initializes the weights to 0.
     * @param Y    Y variables
     * @param X    X variables
     */
    gaussian_crf_factor(const forward_range<vector_variable*>& Y_,
                        copy_ptr<vector_domain>& Xdomain_ptr_);

    /**
     * Constructor.
     * @param ov   optimization_vector defining this factor
     * @param Y    Y variables
     * @param X    X variables
     */
    gaussian_crf_factor(const optimization_vector& ov,
                        const vector_var_vector& Y_,
                        const vector_var_vector& X_);

    /**
     * Constructor.  Takes a linear_regression and constructs the corresponding
     * gaussian_crf_factor.
     */
    gaussian_crf_factor(const linear_regression& lr, const dataset& ds);

    /**
     * Constructor.  Takes a moment_gaussian and constructs the corresponding
     * gaussian_crf_factor.
     * @todo Permit mg to be, e.g., P(Y,X), instead of just P(Y|X).
     */
    explicit gaussian_crf_factor(const moment_gaussian& mg);

    /**
     * Constructor.  Takes a canonical_gaussian and constructs the corresponding
     * gaussian_crf_factor (which is a marginal factor).
     */
    explicit gaussian_crf_factor(const canonical_gaussian& cg);

    /**
     * Constructor from a constant factor.
     */
    explicit gaussian_crf_factor(const constant_factor& cf);

    /**
     * Constructor from a constant.
     */
    explicit gaussian_crf_factor(double c);

    //! @return  vector of output variables in Y for this factor
    const vector_var_vector& output_arg_list() const;

    //! @return  vector of input variables in X for this factor
    const vector_var_vector& input_arg_list() const;

    /**
     * @param print_Y    If true, print Y variables. (default = true)
     * @param print_X    If true, print X variables. (default = true)
     * @param print_vals If true, print factor values. (default = true)
     */
    void print(std::ostream& out, bool print_Y = true, bool print_X = true,
               bool print_vals = true) const;

    //! Return the underlying factor as a moment_gaussian.
    moment_gaussian get_gaussian() const;

    /**
     * Relabels outputs Y, inputs X so that
     * inputs may become outputs (if variable_type = output_variable_type) and
     * outputs may become inputs (if variable_type = input_variable_type).
     * The entire argument set must remain the same, i.e.,
     * union(Y,X) must equal union(new_Y, new_X).
     */
    void relabel_outputs_inputs(const output_domain_type& new_Y,
                                const input_domain_type& new_X);

    // Public methods: Probabilistic queries
    // =========================================================================

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const vector_assignment& a) const;

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const vector_record& r) const;

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const assignment_type& a) const;

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const record_type& r) const;

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     * This is the least efficient condition().
     *
     * @param a  This must assign values to all X in this factor
     *           (but may assign values to any other variables as well).
     * @return  gaussian factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const canonical_gaussian& condition(const vector_assignment& a) const;

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param r Record with values for X in this factor
     *          (which may have values for any other variables as well).
     * @return  gaussian factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const canonical_gaussian& condition(const vector_record& r) const;

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     * This is the most efficient condition().
     *
     * @param x Values for X variables, in the order used by this factor.
     * @return  gaussian factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const canonical_gaussian& condition(const vec& x) const;

    /**
     * If this factor is f(Y_retain, Y_part, X) (not in log space),
     * return a new factor f(Y_retain, X) which represents
     * exp(E[log(f)]) where the expectation is w.r.t. a uniform distribution
     * over all values of Y_part.
     *
     * @param Y_part   Output variables w.r.t. which the expectation is taken
     *                 in log space.
     *
     * @return  This modified factor.
     */
    gaussian_crf_factor&
    partial_expectation_in_log_space(const output_domain_type& Y_part);

    /**
     * If this factor is f(Y_retain, Y_part, X) (not in log space),
     * return a new factor f(Y_retain, X) which represents
     * exp(E[log(f)]) where the expectation is over values of Y_part in
     * records in the given dataset.
     *
     * @param Y_part   Output variables w.r.t. which the expectation is taken
     *                 in log space.
     *
     * @return  This modified factor.
     */
    gaussian_crf_factor&
    partial_expectation_in_log_space(const output_domain_type& Y_part,
                                     const dataset& ds);

    /**
     * If this factor is f(Y_retain, Y_other, X),
     * marginalize out Y_other to get a new factor f(Y_retain, X).
     *
     * @param Y_other   Output variables to marginalize out.
     *
     * @return  This modified factor.
     */
    gaussian_crf_factor& marginalize_out(const output_domain_type& Y_other);

    /**
     * If this factor is f(Y_part, Y_other, X_part, X_other),
     * set it to f(Y_part = y_part, Y_other, X_part = x_part, X_other).
     *
     * @param a Assignment with values for Y_part, X_part in this factor
     *          (which may have values for any other variables as well).
     * @return  This modified factor.
     */
    gaussian_crf_factor& partial_condition(const assignment_type& a,
                                           const output_domain_type& Y_part,
                                           const input_domain_type& X_part);

    /**
     * If this factor is f(Y_part, Y_other, X_part, X_other),
     * set it to f(Y_part = y_part, Y_other, X_part = x_part, X_other).
     *
     * @param r Record with values for Y_part, X_part in this factor
     *          (which may have values for any other variables as well).
     * @return  This modified factor.
     */
    gaussian_crf_factor& partial_condition(const record_type& r,
                                           const output_domain_type& Y_part,
                                           const input_domain_type& X_part);

    /**
     * Returns the empirical expectation of the log of this factor.
     * In particular, if this factor represents P(A|B), then
     * this returns the expected log likelihood of the distribution P(A | B).
     * This uses real-space; i.e., the log of this factor is in log-space.
     */
    double log_expected_value(const dataset& ds) const;

    //! implements Factor::combine_in
    gaussian_crf_factor&
    combine_in(const gaussian_crf_factor& other, op_type op);

    //! combines a constant factor into this factor
    gaussian_crf_factor&
    combine_in(const constant_factor& other, op_type op);

    //! Combine with constant factor via op(f, *this).
    //! @return This modified factor.
//  gaussian_crf_factor& combine_in_left(const constant_factor& cf, op_type op);

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

    /**
     * When called, this fixes this factor to use records of this type very
     * efficiently (setting fixed_records_ = true).
     * This option MUST be turned off before using this factor with records
     * with different variable orderings!
     */
    void fix_records(const vector_record& r) {
      r.vector_indices(Y_indices_, Y_);
      r.vector_indices(X_indices_, X_);
      fixed_records_ = true;
    }

    /**
     * This turns off fixed_records_, allowing this factor to be used with
     * any records.
     */
    void unfix_records() {
      fixed_records_ = false;
    }

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space.
    const optimization_vector& weights() const {
      return ov;
    }

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space.
    optimization_vector& weights() {
      return ov;
    }

    // Public: Learning methods from learnable_crf_factor interface
    // =========================================================================

    //! Adds the gradient of the log of this factor w.r.t. the weights,
    //! evaluated at the given datapoint with the current weights.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param w      Weight by which to multiply the added values.
    void add_gradient(optimization_vector& grad, const vector_record& r,
                      double w) const;

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
     * @tparam YFactor  Factor type for a distribution over Y variables.
     */
    void add_expected_gradient(optimization_vector& grad,
                               const vector_record& r,
                               const canonical_gaussian& fy,
                               double w = 1) const;

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
     * @tparam YFactor  Factor type for a distribution over Y variables.
     */
    void add_expected_gradient(optimization_vector& grad,
                               const vector_record& r,
                               const moment_gaussian& fy, double w = 1) const;

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void
    add_combined_gradient(optimization_vector& grad, const vector_record& r,
                          const canonical_gaussian& fy, double w = 1.) const;

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void
    add_combined_gradient(optimization_vector& grad, const vector_record& r,
                          const moment_gaussian& fy, double w = 1.) const;

    /**
     * Adds the diagonal of the Hessian of the log of this factor w.r.t. the
     * weights, evaluated at the given datapoint with the current weights.
     * @param hessian Pre-allocated vector to which to add the hessian.
     * @param r       Datapoint.
     * @param w       Weight by which to multiply the added values.
     */
    void
    add_hessian_diag(optimization_vector& hessian, const vector_record& r,
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
                              const vector_record& r,
                              const canonical_gaussian& fy, double w) const;

    void
    add_expected_hessian_diag(optimization_vector& hessian,
                              const vector_record& r,
                              const moment_gaussian& fy, double w) const;

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
                                  const vector_record& r,
                                  const canonical_gaussian& fy, double w) const;

    void
    add_expected_squared_gradient(optimization_vector& sqrgrad,
                                  const vector_record& r,
                                  const moment_gaussian& fy, double w) const;

    /**
     * Returns the regularization penalty for the current weights and
     * the given regularization parameters.
     */
    double regularization_penalty(const regularization_type& reg) const;

    /**
     * Adds the gradient of the regularization term for the current weights
     * and the given regularization parameters.
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

    // Public methods: Operators
    // =========================================================================

    /**
     * Multiplication with another factor.
     * @param other  Factor f(Y_other,X_other), where Y_other must be disjoint
     *               from this factor's Y and X_other disjoint from this
     *               factor's X.
     */
    gaussian_crf_factor& operator*=(const gaussian_crf_factor& other);

    // Protected data and methods
    // =========================================================================
  protected:

    optimization_vector ov;

    //! Y variables
    vector_var_vector Y_;

    //! X variables
    vector_var_vector X_;

    //! If true, then whenever methods taking records are called,
    //! it is assumed that records of exactly the same type are being given.
    //! This will then use the stored indices Y_indices_, X_indices_ to
    //! access data in the records.
    bool fixed_records_;

    //! See fixed_records_; Y_indices_ = indices of Y_ in records.
    ivec Y_indices_;

    //! See fixed_records_; X_indices_ = indices of X_ in records.
    ivec X_indices_;

    //! Temporary used to avoid reallocation for conditioning.
    //! If this CRF factor has no arguments (i.e., is a constant factor),
    //! then this stored its value. // TO DO: ELIMINATE THIS HACK.
    mutable canonical_gaussian conditioned_f;

    //! Get Y,X values from record.
    void get_yx_values(const vector_record& r, vec& y, vec& x) const {
      if (fixed_records_) {
        const vec& rvec = r.vector();
        y = rvec(Y_indices_);
        x = rvec(X_indices_);
      } else {
        r.vector_values(y, Y_);
        r.vector_values(x, X_);
      }
    }

    //! Get Y values from record.
    void get_y_values(const vector_record& r, vec& y) const {
      if (fixed_records_) {
        const vec& rvec = r.vector();
        y = rvec(Y_indices_);
      } else {
        r.vector_values(y, Y_);
      }
    }

    //! Get Y,X values from record.
    void get_x_values(const vector_record& r, vec& x) const {
      if (fixed_records_) {
        const vec& rvec = r.vector();
        x = rvec(X_indices_);
      } else {
        r.vector_values(x, X_);
      }
    }

  };  // class gaussian_crf_factor

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_GAUSSIAN_CRF_FACTOR_HPP
