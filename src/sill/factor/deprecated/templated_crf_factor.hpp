#ifndef SILL_TEMPLATED_CRF_FACTOR_HPP
#define SILL_TEMPLATED_CRF_FACTOR_HPP

#include <sill/factor/concepts.hpp>
#include <sill/factor/crf/learnable_crf_factor.hpp>
#include <sill/learning/dataset_old/record_conversions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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
    typedef typename base::input_variable_type   input_variable_type;
    typedef typename base::input_domain_type     input_domain_type;
    typedef typename base::input_assignment_type input_assignment_type;
    typedef typename base::input_var_vector_type input_var_vector_type;
    typedef typename base::input_record_type     input_record_type;
    typedef typename base::output_variable_type   output_variable_type;
    typedef typename base::output_domain_type     output_domain_type;
    typedef typename base::output_assignment_type output_assignment_type;
    typedef typename base::output_var_vector_type output_var_vector_type;
    typedef typename base::output_record_type     output_record_type;
    typedef typename base::variable_type   variable_type;
    typedef typename base::domain_type     domain_type;
    typedef typename base::assignment_type assignment_type;
    typedef typename base::var_vector_type var_vector_type;
    typedef typename base::record_type     record_type;
    typedef typename base::result_type          result_type;
    typedef typename base::output_factor_type   output_factor_type;
    typedef typename base::optimization_vector  optimization_vector;
    typedef typename base::regularization_type  regularization_type;


    typedef std::map<input_variable_type, input_variable_type> input_var_map_type;
    typedef std::map<output_variable_type, output_variable_type> output_var_map_type;
    typedef std::map<variable_type, variable_type> var_map_type;

    typedef typename F::la_type la_type;

    //! Parameters used for learn_crf_factor.
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
      : base(), factor_ptr_(new F()), fixed_records_(false),
        relabeled_args(false) { }

    /**
     * Constructor which creates a template of the given factor.
     * This template uses the same Y,X variables as in the given factor.
     *
     * @param factor_ptr_  Factor being templated.
     *                      Note: This class uses this same pointer,
     *                      rather than doing a deep copy.
     */
    templated_crf_factor(boost::shared_ptr<F> factor_ptr_)
      : base(*factor_ptr_), factor_ptr_(factor_ptr_), fixed_records_(false),
        relabeled_args(false) {
      init();
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
    templated_crf_factor(boost::shared_ptr<F> factor_ptr_,
                         const output_var_map_type& Yvarmap,
                         const input_var_map_type& Xvarmap)
      : base(), factor_ptr_(factor_ptr_), fixed_records_(false),
        relabeled_args(false) {
      init(Yvarmap,Xvarmap);
    }

    /**
     * Constructor a factor with default parameters.
     *
     * NOTE: This creates a new base factor since it does not have another
     *       factor from which to create a template.
     *
     * @param Y   Output arguments.
     * @param X   Input arguments.
     */
    templated_crf_factor(const output_domain_type& Y_,
                         const input_domain_type& X_)
      : base(Y_, X_), factor_ptr_(new F(Y_,X_)), fixed_records_(false),
        relabeled_args(false) {
      init();
    }

    /**
     * Copy constructor.
     * WARNING: The new copy uses the same shared base factor;
     *          it does NOT do a deep copy.
     * @todo Decide if this should or should not use the same shared base
     *       factor.  Where is this used?
     */
    templated_crf_factor(const templated_crf_factor& other)
      : base(other), factor_ptr_(other.factor_ptr_),
        base_record(other.base_record),
        base_input_record(other.base_input_record),
        vmap_base2this(other.vmap_base2this),
        vmap_this2base(other.vmap_this2base),  tmp_factor(other.tmp_factor),
        fixed_records_(other.fixed_records_),
        relabeled_args(other.relabeled_args),
        base_newY(other.base_newY), base_newX(other.base_newX),
        base_oldY(other.base_oldY), base_oldX(other.base_oldX){ }

    /**
     * Assignment operator.
     * WARNING: The new copy uses the same shared base factor as 'other';
     *          it does NOT do a deep copy.
     * @todo Decide if this should or should not use the same shared base
     *       factor.  Where is this used?
     */
    templated_crf_factor& operator=(const templated_crf_factor& other) {
      base::operator=(other);
      factor_ptr_ = other.factor_ptr_;
      base_record = other.base_record;
      base_input_record = other.base_input_record;
      vmap_base2this = other.vmap_base2this;
      vmap_this2base = other.vmap_this2base;
      tmp_factor = other.tmp_factor;
      fixed_records_ = other.fixed_records_;
      relabeled_args = other.relabeled_args;
      base_newY = other.base_newY;
      base_newX = other.base_newX;
      base_oldY = other.base_oldY;
      base_oldX = other.base_oldX;
      return *this;
    }

    // Getters and helpers
    // =========================================================================

    using base::output_arguments;
    using base::input_arguments;

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

    using base::relabel_outputs_inputs;

    /**
     * Relabels outputs Y, inputs X so that
     * inputs may become outputs (if variable_type = output_variable_type) and
     * outputs may become inputs (if variable_type = input_variable_type).
     * The entire argument set must remain the same, i.e.,
     * union(Y,X) must be a subset of union(new_Y, new_X).
     */
    void relabel_outputs_inputs(const output_domain_type& new_Y,
                                const input_domain_type& new_X) {
      assert(factor_ptr_);
      unfix_records();
      // Adjust templated factor's arguments.
      domain_type old_args(output_arguments());
      old_args.insert(input_arguments().begin(), input_arguments().end());
      Ydomain_ = set_intersect(old_args,new_Y);
      Xdomain_ptr_->operator=(set_intersect(old_args,new_X));
      if (!set_disjoint(output_arguments(), input_arguments())) {
        throw std::invalid_argument
          (std::string("templated_crf_factor::relabel_outputs_inputs given") +
           " new_Y,new_X which were not disjoint.");
      }
      if (output_arguments().size() + input_arguments().size()
          != old_args.size()) {
        throw std::invalid_argument
          (std::string("templated_crf_factor::relabel_outputs_inputs given ") +
           "new_Y,new_X whose union did not include the union of the old Y,X.");
      }
      // Pre-compute base factor's arguments.
      base_newY.clear();
      base_newX.clear();
      foreach(output_variable_type v, output_arguments())
        base_newY.insert(safe_get(vmap_this2base,v));
      foreach(input_variable_type v, input_arguments())
        base_newX.insert(safe_get(vmap_this2base,v));
      base_oldY = factor_ptr_->output_arguments();
      base_oldX = factor_ptr_->input_arguments();
      // Set base_input_record.
      input_var_vector_type base_input_vars(base_newX.begin(), base_newX.end());
      base_input_record = input_record_type(base_input_vars);

      relabeled_args = true;
    } // relabel_outputs_inputs

    //! Get the base factor.
    const F& base_factor() const {
      assert(factor_ptr_);
      return *factor_ptr_;
    }

    // Public methods: Probabilistic queries
    // =========================================================================

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const assignment_type& a) const {
      base_record.copy_from_assignment_mapped(a, vmap_base2this);
      return v_sub();
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const record_type& r) const {
      base_record.copy_from_record_mapped(r, vmap_base2this);
      return v_sub();
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const assignment_type& a) const {
      base_record.copy_from_assignment_mapped(a, vmap_base2this);
      return logv_sub();
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const record_type& r) const {
      base_record.copy_from_record_mapped(r, vmap_base2this);
      return logv_sub();
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
      base_input_record.copy_from_assignment_mapped(a, vmap_base2this);
      condition_sub();
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
      base_input_record.copy_from_record_mapped(r, vmap_base2this);
      condition_sub();
      return tmp_factor;
    }

    /**
     * Returns the empirical expectation of the log of this factor.
     * In particular, if this factor represents P(A|B), then
     * this returns the expected log likelihood of the distribution P(A | B).
     * This uses real-space; i.e., the log of this factor is in log-space.
     */
    double log_expected_value(const dataset<la_type>& ds) const {
      double val(0);
      output_factor_type tmp_fctr;
      double total_ds_weight(0);
      size_t i(0);
      foreach(const record_type& r, ds.records()) {
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
      base_record.copy_from_record_mapped(r, vmap_base2this);
      factor_ptr_->fix_records(base_record);
      fixed_records_ = true;
    }

    /**
     * This turns off fixed_records_, allowing this factor to be used with
     * any records.
     */
    void unfix_records() {
      factor_ptr_->unfix_records();
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
      base_record.copy_from_record_mapped(r, vmap_base2this);
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        factor_ptr_->add_gradient(grad, base_record, w);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
      } else {
        factor_ptr_->add_gradient(grad, base_record, w);
      }
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
      base_input_record.copy_from_record_mapped(r, vmap_base2this);
      tmp_factor = fy;
      tmp_factor.subst_args(vmap_this2base);
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        factor_ptr_->add_expected_gradient(grad,base_input_record,tmp_factor,w);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
      } else {
        factor_ptr_->add_expected_gradient(grad,base_input_record,tmp_factor,w);
      }
    }

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void
    add_combined_gradient(optimization_vector& grad, const record_type& r,
                          const output_factor_type& fy, double w = 1.) const {
      base_record.copy_from_record_mapped(r, vmap_base2this);
      tmp_factor = fy;
      tmp_factor.subst_args(vmap_this2base);
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        factor_ptr_->add_combined_gradient(grad, base_record, tmp_factor, w);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
      } else {
        factor_ptr_->add_combined_gradient(grad, base_record, tmp_factor, w);
      }
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
      base_record.copy_from_record_mapped(r, vmap_base2this);
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        factor_ptr_->add_hessian_diag(hessian, base_record, w);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
      } else {
        factor_ptr_->add_hessian_diag(hessian, base_record, w);
      }
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
      base_input_record.copy_from_record_mapped(r, vmap_base2this);
      tmp_factor = fy;
      tmp_factor.subst_args(vmap_this2base);

      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        factor_ptr_->add_expected_hessian_diag(hessian, base_input_record,
                                               tmp_factor, w);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
      } else {
        factor_ptr_->add_expected_hessian_diag(hessian, base_input_record,
                                               tmp_factor, w);
      }
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
      base_input_record.copy_from_record_mapped(r, vmap_base2this);
      tmp_factor = fy;
      tmp_factor.subst_args(vmap_this2base);
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        factor_ptr_->add_expected_squared_gradient(sqrgrad, base_input_record,
                                                   tmp_factor, w);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
      } else {
        factor_ptr_->add_expected_squared_gradient(sqrgrad, base_input_record,
                                                   tmp_factor, w);
      }
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
    var_map_type vmap_base2this;

    //! Mapping: Vars in this factor instance --> Vars in the base factor
    var_map_type vmap_this2base;

    //! Temp used to hold factors returned by the base,
    //! but with variables mapped to this instance's variables.
    mutable output_factor_type tmp_factor;

    //! True if fix_records() has been called.
    bool fixed_records_;

    //! If set, then base_newY, base_newX are set too.
    bool relabeled_args;

    //! Stores relabeled outputs/inputs for base factor.
    output_domain_type base_newY;
    input_domain_type base_newX;

    //! Stores original outputs/inputs for base factor.
    output_domain_type base_oldY;
    input_domain_type base_oldX;

    // Private methods
    // =========================================================================

    double v_sub() const {
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        double val = factor_ptr_->v(base_record);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
        return val;
      } else {
        return factor_ptr_->v(base_record);
      }
    }

    double logv_sub() const {
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        double val = factor_ptr_->logv(base_record);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
        return val;
      } else {
        return factor_ptr_->logv(base_record);
      }
    }

    void condition_sub() const {
      if (relabeled_args) {
        factor_ptr_->relabel_outputs_inputs(base_newY, base_newX);
        tmp_factor = factor_ptr_->condition(base_input_record);
        factor_ptr_->relabel_outputs_inputs(base_oldY, base_oldX);
      } else {
        tmp_factor = factor_ptr_->condition(base_input_record);
      }
      tmp_factor.subst_args(vmap_base2this);
    }

    //! Initialize with identity variable map.
    //! Given: factor_ptr_ is set
    //! Init: vmap_base2this, base_record, base_input_record
    void init() {
      assert(factor_ptr_);
      var_vector_type base_vars;
      input_var_vector_type base_input_vars;
      foreach(output_variable_type v, factor_ptr_->output_arguments()) {
        base_vars.push_back(v);
        vmap_base2this[v] = v;
        vmap_this2base[v] = v;
      }
      foreach(input_variable_type v, factor_ptr_->input_arguments()) {
        base_vars.push_back(v);
        base_input_vars.push_back(v);
        vmap_base2this[v] = v;
        vmap_this2base[v] = v;
      }
      base_record = record_type(base_vars);
      base_input_record = input_record_type(base_input_vars);
    }

    //! Initialize with the given variable maps.
    //! Given: factor_ptr_ is set
    //! Init: vmap_base2this, base_record, base_input_record
    void init(const output_var_map_type& Yvarmap,
              const input_var_map_type& Xvarmap) {
      assert(factor_ptr_);
      var_vector_type base_vars;
      input_var_vector_type base_input_vars;
      foreach(output_variable_type v, factor_ptr_->output_arguments()) {
        output_variable_type my_v = safe_get(Yvarmap, v);
        Ydomain_.insert(my_v);
        base_vars.push_back(v);
        vmap_base2this[v] = my_v;
        vmap_this2base[my_v] = v;
      }
      foreach(input_variable_type v, factor_ptr_->input_arguments()) {
        input_variable_type my_v = safe_get(Xvarmap, v);
        Xdomain_ptr_->insert(my_v);
        base_vars.push_back(v);
        base_input_vars.push_back(v);
        vmap_base2this[v] = my_v;
        vmap_this2base[my_v] = v;
      }
      base_record = record_type(base_vars);
      base_input_record = input_record_type(base_input_vars);
    }

  };  // class templated_crf_factor

};  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_TEMPLATED_CRF_FACTOR_HPP
