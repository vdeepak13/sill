#ifndef PRL_TEMPLATED_CRF_FACTOR_HPP
#define PRL_TEMPLATED_CRF_FACTOR_HPP

#include <prl/factor/concepts.hpp>
#include <prl/factor/learnable_crf_factor.hpp>
#include <prl/learning/dataset/record_conversions.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * CRF factor representing P(Y | X).
   * This permits templated models, or models with tied parameters
   * (where multiple factors within a crf_model share the same parameters).
   *
   * This satisfies the LearnableCRFfactor concept.
   *
   * @tparam F  Factor type being templated (e.g., table_crf_factor).
   *            This must inherit from learnable_crf_factor.
   *
   * \ingroup factor
   * @author Joseph Bradley
   */
  template <typename F>
  class templated_crf_factor
    : public learnable_crf_factor<typename F::input_variable_type,
                                  typename F::output_factor_type,
                                  typename F::optimization_vector,
                                  F::regularization_type::nlambdas> {

    // Public types
    // =========================================================================
  public:

    //! Base class
    typedef
    learnable_crf_factor<typename F::input_variable_type,
                         typename F::output_factor_type,
                         typename F::optimization_vector,
                         F::regularization_type::nlambdas>
    base;

    typedef F factor_type;

    // Import types from base.
    typedef typename base::input_variable_type input_variable_type;
    typedef typename base::input_domain_type input_domain_type;
    typedef typename base::input_assignment_type input_assignment_type;
    typedef typename base::input_record_type input_record_type;
    typedef typename base::output_variable_type output_variable_type;
    typedef typename base::output_domain_type output_domain_type;
    typedef typename base::output_assignment_type output_assignment_type;
    typedef typename base::output_record_type output_record_type;
    typedef typename base::variable_type variable_type;
    typedef typename base::domain_type domain_type;
    typedef typename base::assignment_type assignment_type;
    typedef typename base::record_type record_type;
    typedef typename base::result_type result_type;
    typedef typename base::output_factor_type output_factor_type;
    typedef typename base::optimization_vector optimization_vector;
    typedef typename base::regularization_type regularization_type;

    //! Parameters used for learn_crf_factor().
    typedef typename F::parameters parameters;

  private:

    using base::Ydomain_;
    using base::Xdomain_ptr_;
    using base::fixed_value_;

    // Constructors and destructor
    // =========================================================================
  public:

    //! Default constructor.
    templated_crf_factor()
      : base(), factor_ptr_(new F()), fixed_records_(false) { }

    /**
     * Constructor which creates a template of the given factor.
     * This template uses the same Y,X variables as in the given factor.
     *
     * @param factor_ptr_  Factor being templated.
     *                      Note: This class uses this same pointer,
     *                      rather than doing a deep copy.
     */
    templated_crf_factor
    (boost::shared_ptr<F> factor_ptr_)
      : base(*factor_ptr_), factor_ptr_(factor_ptr_), fixed_records_(false) {
      typename variable_type_group<variable_type>::var_vector_type base_vars;
      typename variable_type_group<input_variable_type>::var_vector_type
        base_input_vars;
      foreach(output_variable_type* v, factor_ptr_->output_arguments()) {
        base_vars.push_back(v);
        vmap_base2this[v] = v;
      }
      foreach(input_variable_type* v, factor_ptr_->input_arguments()) {
        base_vars.push_back(v);
        base_input_vars.push_back(v);
        vmap_base2this[v] = v;
      }
      base_record = record_type(base_vars);
      base_input_record = input_record_type(base_input_vars);
    }

    /**
     * Constructor which creates a template of the given factor.
     * This template uses the Y,X variables given by the mappings,
     * which may be different than the ones in the given factor.
     *
     * @param factor_ptr_  Factor being templated.
     *                      Note: This class uses this same pointer,
     *                      rather than doing a deep copy.
     * @param Yvarmap      Mapping from Y variables in factor_ptr_ to
     *                      variables to be used for this instance of
     *                      the templated factor.
     *                      (default = identity map)
     * @param Xvarmap      Mapping from X variables in factor_ptr_ to
     *                      variables to be used for this instance of
     *                      the templated factor.
     *                      (default = identity map)
     */
    templated_crf_factor
    (boost::shared_ptr<F> factor_ptr_,
     const typename variable_type_group<output_variable_type>::var_map_type&
     Yvarmap,
     const typename variable_type_group<input_variable_type>::var_map_type&
     Xvarmap)
      : base(), factor_ptr_(factor_ptr_), fixed_records_(false) {
      assert(factor_ptr_);
      typename variable_type_group<variable_type>::var_vector_type base_vars;
      typename variable_type_group<input_variable_type>::var_vector_type
        base_input_vars;
      foreach(output_variable_type* v, factor_ptr_->output_arguments()) {
        output_variable_type* my_v = safe_get(Yvarmap, v);
        Ydomain_.insert(my_v);
        base_vars.push_back(v);
        vmap_base2this[v] = my_v;
      }
      foreach(input_variable_type* v, factor_ptr_->input_arguments()) {
        input_variable_type* my_v = safe_get(Xvarmap, v);
        Xdomain_ptr_->insert(my_v);
        base_vars.push_back(v);
        base_input_vars.push_back(v);
        vmap_base2this[v] = my_v;
      }
      base_record = record_type(base_vars);
      base_input_record = input_record_type(base_input_vars);
    }

    /**
     * Copy constructor.
     * WARNING: The new copy uses the same shared base factor;
     *          it does NOT do a deep copy.
     */
    templated_crf_factor(const templated_crf_factor& other)
      : base(other), factor_ptr_(other.factor_ptr_),
        base_record(other.base_record),
        base_input_record(other.base_input_record),
        vmap_base2this(other.vmap_base2this), tmp_factor(other.tmp_factor),
        fixed_records_(other.fixed_records_) { }

    /**
     * Assignment operator.
     * WARNING: The new copy uses the same shared base factor as 'other';
     *          it does NOT do a deep copy.
     */
    templated_crf_factor& operator=(const templated_crf_factor& other) {
      base::operator=(other);
      factor_ptr_ = other.factor_ptr_;
      base_record = other.base_record;
      base_input_record = other.base_input_record;
      vmap_base2this = other.vmap_base2this;
      tmp_factor = other.tmp_factor;
      fixed_records_ = other.fixed_records_;
      return *this;
    }

    // Getters and helpers
    // =========================================================================

    /**
     * @param print_Y    If true, print Y variables. (default = true)
     * @param print_X    If true, print X variables. (default = true)
     * @param print_vals If true, print factor values. (default = true)
     */
    void print(std::ostream& out, bool print_Y = true, bool print_X = true,
               bool print_vals = true) const {
      out << "F[";
      if (print_Y)
        out << Ydomain_;
      else
        out << "*";
      out << ", ";
      if (print_X)
        out << (*Xdomain_ptr_);
      else
        out << "*";
      if (print_vals) {
        out << "; base factor: " << (*factor_ptr_);
      }
      out << "]\n";
      out << std::flush;
    }

    // Public methods: Probabilistic queries
    // =========================================================================

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const assignment_type& a) const {
      fill_record_with_assignment(base_record, a, vmap_base2this);
      return factor_ptr_->v(base_record);
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const record_type& r) const {
      fill_record_with_record(base_record, r, vmap_base2this);
      return factor_ptr_->v(base_record);
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const assignment_type& a) const {
      fill_record_with_assignment(base_record, a, vmap_base2this);
      return factor_ptr_->logv(base_record);
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const record_type& r) const {
      fill_record_with_record(base_record, r, vmap_base2this);
      return factor_ptr_->logv(base_record);
    }

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
    const output_factor_type& condition(const input_assignment_type& a) const {
      fill_record_with_assignment(base_input_record, a, vmap_base2this);
      tmp_factor = factor_ptr_->condition(base_input_record);
      tmp_factor.subst_args(vmap_base2this);
      return tmp_factor;
    }

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param r Record with values for X in this factor
     *          (which may have values for any other variables as well).
     * @return  gaussian factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const output_factor_type& condition(const input_record_type& r) const {
      fill_record_with_record(base_input_record, r, vmap_base2this);
      tmp_factor = factor_ptr_->condition(base_input_record);
      tmp_factor.subst_args(vmap_base2this);
      return tmp_factor;
    }

    /**
     * Returns the empirical expectation of the log of this factor.
     * In particular, if this factor represents P(A|B), then
     * this returns the expected log likelihood of the distribution P(A | B).
     * This uses real-space; i.e., the log of this factor is in log-space.
     */
    double log_expected_value(const dataset& ds) const {
      double val(0);
      output_factor_type tmp_fctr;
      double total_ds_weight(0);
      size_t i(0);
      foreach(const record& r, ds.records()) {
        val += ds.weight(i) * logv(r);
        total_ds_weight += ds.weight(i);
        ++i;
      }
      assert(total_ds_weight > 0);
      return (val / total_ds_weight);        
    }

    // Public: Learning-related methods from crf_factor interface
    // =========================================================================

    //! @return  true iff the data is stored in log-space
    bool log_space() const {
      return factor_ptr_->log_space();
    }

    //! Tries to change this factor's internal representation to log-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already log-space
    bool convert_to_log_space() {
      return factor_ptr_->convert_to_log_space();
    }

    //! Tries to change this factor's internal representation to real-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already real-space
    bool convert_to_real_space() {
      return factor_ptr_->convert_to_real_space();
    }

    /**
     * When called, this fixes this factor to use records of this type very
     * efficiently (setting fixed_records_ = true).
     * This option MUST be turned off before using this factor with records
     * with different variable orderings!
     */
    void fix_records(const record_type& r) {
      assert(false); // TO DO
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
      return factor_ptr_->weights();
    }

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space.
    optimization_vector& weights() {
      return factor_ptr_->weights();
    }

    // Public: Learning methods from learnable_crf_factor interface
    // =========================================================================

    //! Adds the gradient of the log of this factor w.r.t. the weights,
    //! evaluated at the given datapoint with the current weights.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param w      Weight by which to multiply the added values.
    void add_gradient(optimization_vector& grad, const record_type& r,
                      double w) const {
      fill_record_with_record(base_record, r, vmap_base2this);
      factor_ptr_->add_gradient(grad, base_record, w);
    }

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
                               const input_record_type& r,
                               const output_factor_type& fy,
                               double w = 1) const {
      fill_record_with_record(base_input_record, r, vmap_base2this);
      factor_ptr_->add_expected_gradient(grad, base_input_record, fy, w);
    }

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void
    add_combined_gradient(optimization_vector& grad, const record_type& r,
                          const output_factor_type& fy, double w = 1.) const {
      fill_record_with_record(base_record, r, vmap_base2this);
      factor_ptr_->add_combined_gradient(grad, base_record, fy, w);
    }

    /**
     * Adds the diagonal of the Hessian of the log of this factor w.r.t. the
     * weights, evaluated at the given datapoint with the current weights.
     * @param hessian Pre-allocated vector to which to add the hessian.
     * @param r       Datapoint.
     * @param w       Weight by which to multiply the added values.
     */
    void
    add_hessian_diag(optimization_vector& hessian, const record_type& r,
                     double w) const {
      fill_record_with_record(base_record, r, vmap_base2this);
      factor_ptr_->add_hessian_diag(hessian, base_record, w);
    }

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
                              const output_factor_type& fy, double w) const {
      fill_record_with_record(base_input_record, r, vmap_base2this);
      factor_ptr_->add_expected_hessian_diag(hessian, base_input_record, fy, w);
    }

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
    add_expected_squared_gradient
    (optimization_vector& sqrgrad, const input_record_type& r,
     const output_factor_type& fy, double w) const {
      fill_record_with_record(base_input_record, r, vmap_base2this);
      factor_ptr_->add_expected_squared_gradient(sqrgrad, base_input_record,
                                                 fy, w);
    }

    /**
     * Returns the regularization penalty for the current weights and
     * the given regularization parameters.
     */
    double regularization_penalty(const regularization_type& reg) const {
      return factor_ptr_->regularization_penalty(reg);
    }

    /**
     * Adds the gradient of the regularization term for the current weights
     * and the given regularization parameters.
     */
    void add_regularization_gradient(optimization_vector& grad,
                                     const regularization_type& reg,
                                     double w) const {
      return factor_ptr_->add_regularization_gradient(grad, reg, w);
    }

    /**
     * Adds the diagonal of the Hessian of the regularization term for the
     * current weights and the given regularization parameters.
     */
    void add_regularization_hessian_diag(optimization_vector& hd,
                                         const regularization_type& reg,
                                         double w) const {
      return factor_ptr_->add_regularization_hessian_diag(hd, reg, w);
    }

    // Private data
    // =========================================================================
  private:

    boost::shared_ptr<F> factor_ptr_;

    //! Record passed to base factor.
    //! This uses the base factor's Y,X variables.
    mutable record_type base_record;

    //! Record passed to base factor, but with only the X variables.
    //! This uses the base factor's X variables.
    mutable record_type base_input_record;

    //! Mapping: Vars in the base factor --> Vars in this factor instance
    typename variable_type_group<variable_type>::var_map_type vmap_base2this;

    //! Temp used to hold factors returned by the base,
    //! but with variables mapped to this instance's variables.
    mutable output_factor_type tmp_factor;

    //! True if fix_records() has been called.
    bool fixed_records_;

    // Private methods
    // =========================================================================

  };  // class templated_crf_factor

};  // namespace prl

#include <prl/macros_undef.hpp>

#endif // PRL_TEMPLATED_CRF_FACTOR_HPP
