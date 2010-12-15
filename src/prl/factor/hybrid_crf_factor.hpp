#ifndef PRL_HYBRID_CRF_FACTOR_HPP
#define PRL_HYBRID_CRF_FACTOR_HPP

#include <prl/base/finite_assignment_iterator.hpp>
#include <prl/factor/concepts.hpp>
#include <prl/factor/learnable_crf_factor.hpp>
#include <prl/optimization/hybrid_opt_vector.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * CRF factor representing P(Y | X1, X2), where X2 are finite variables.
   * For every assignment X2 = x2, this maintains a separate factor representing
   *   P(Y | X1, X2=x2).
   * This permits, e.g., hybrid table-Gaussian factors, where the sub-factors
   * are gaussian_crf_factor types.
   *
   * This satisfies the LearnableCRFfactor concept.
   *
   * Note: Learning is done separately for each sub-factor,
   *       so the regularization and lambda info are the same as those of
   *       the sub-factor type.
   *       Moreover, the corresponding learn_crf_factor_cv method chooses
   *       regularization independently for each sub-factor.
   *
   * @tparam F  Sub-factor type.  This must inherit from learnable_crf_factor.
   *
   * \ingroup factor
   * @author Joseph Bradley
   *
   * @todo This currently assumes that all factors are stored in log space
   *       iff the first factor is.  Add checks to ensure this.
   */
  template <typename F>
  class hybrid_crf_factor
    : public learnable_crf_factor<typename variable_type_union
                                  <finite_variable,
                                   typename F::input_variable_type>::union_type,
                                  typename F::output_factor_type,
                                  hybrid_opt_vector
                                  <typename F::optimization_vector>,
                                  F::regularization_type::nlambdas> {

    // Public types
    // =========================================================================
  public:

    //! Base class
    typedef
    learnable_crf_factor<typename variable_type_union
                         <finite_variable,
                          typename F::input_variable_type>::union_type,
                         typename F::output_factor_type,
                         hybrid_opt_vector<typename F::optimization_vector>,
                         F::regularization_type::nlambdas>
    base;

    typedef F sub_factor_type;

    typedef typename base::input_variable_type input_variable_type;
    typedef typename base::input_domain_type input_domain_type;
    typedef typename base::input_assignment_type input_assignment_type;
    typedef typename base::input_record_type input_record_type;

    typedef typename base::assignment_type assignment_type;
    typedef typename base::record_type record_type;

    typedef typename base::output_domain_type output_domain_type;
    typedef typename base::output_factor_type output_factor_type;

    typedef typename base::optimization_vector optimization_vector;
    typedef typename base::regularization_type regularization_type;

    //! Parameters used for learn_crf_factor().
    struct parameters
      : public F::parameters {

      //! Variables X2 modeled by this hybrid (not by the sub-factors).
      finite_domain hcf_x2;

      parameters(const typename F::parameters& sub_parameters)
        : F::parameters(sub_parameters) { }

      bool valid() const {
        return F::parameters::valid();
      }

    }; // struct parameters

  private:

    using base::Ydomain_;
    using base::Xdomain_ptr_;
    using base::fixed_value_;

    // Constructors and destructor
    // =========================================================================
  public:

    //! Default constructor.
    hybrid_crf_factor()
      : base(), fixed_records_(false) { }

    /**
     * Constructor used by learn_crf_factor.
     * WARNING: This class owns subfactor_ptrs_ from here on.
     *          Do not free them outside of this class.
     */
    hybrid_crf_factor
    (const output_domain_type& Y_, copy_ptr<input_domain_type> X_ptr_,
     const std::vector<F*> subfactor_ptrs_,
     const finite_var_vector& x2_vec, const std::vector<size_t> x2_multipliers)
      : base(Y_, X_ptr_), subfactor_ptrs_(subfactor_ptrs_), x2_vec(x2_vec),
        x2_multipliers(x2_multipliers), fixed_records_(false) {
      assert(x2_vec.size() == x2_multipliers.size());
      assert(subfactor_ptrs_.size() == num_assignments(make_domain(x2_vec)));
      set_ov();
    }

    /**
     * Constructor which initializes the weights to 0.
     * @param Y     Y variables
     * @param X1    X1 variables (modeled by the sub-factors)
     * @param X2    X2 variables (modeled as a table of sub-factors)
     */
    hybrid_crf_factor(const output_domain_type& Y_,
                      copy_ptr<input_domain_type>& X_ptr_,
                      const finite_domain& X2_)
      : base(Y_, X_ptr_),
        subfactor_ptrs_(num_assignments(X2_), NULL),
        x2_vec(X2_.begin(), X2_.end()),
        x2_multipliers(compute_multipliers(x2_vec)),
        fixed_records_(false) {
      assert(X_ptr_);
      assert(includes(*X_ptr_, X2_));
      copy_ptr<typename F::input_domain_type>
        sub_X_ptr(new typename F::input_domain_type());
      convert_domain<input_variable_type, typename F::input_variable_type>
        (set_difference(*X_ptr_, X2_), *sub_X_ptr);
      for (size_t i(0); i < num_assignments(X2_); ++i)
        subfactor_ptrs_[i] = new F(Y_, sub_X_ptr);
      set_ov();
    }

    //! Copy constructor.
    hybrid_crf_factor(const hybrid_crf_factor& other)
      : base(other), subfactor_ptrs_(other.subfactor_ptrs_.size(), NULL),
        x2_vec(other.x2_vec), x2_multipliers(other.x2_multipliers),
        fixed_records_(other.fixed_records_),
        x2_fixed_indices(other.x2_fixed_indices) {
      std::vector<typename F::optimization_vector*>
        sub_ov_ptrs(subfactor_ptrs_.size(), NULL);
      for (size_t i(0); i < subfactor_ptrs_.size(); ++i) {
        subfactor_ptrs_[i] = new F(*(other.subfactor_ptrs_[i]));
        sub_ov_ptrs[i] = &(subfactor_ptrs_[i]->weights());
      }
      ov = optimization_vector(sub_ov_ptrs);
    }

    //! Assignment operator.
    hybrid_crf_factor& operator=(const hybrid_crf_factor& other) {
      Ydomain_ = other.Ydomain_;
      Xdomain_ptr_ = other.Xdomain_ptr_;
      fixed_value_ = other.fixed_value_;
      // Free old data
      foreach(F* subfptr, subfactor_ptrs_)
        delete subfptr;
      // Copy new data
      subfactor_ptrs_.resize(other.subfactor_ptrs_.size());
      x2_vec = other.x2_vec;
      x2_multipliers = other.x2_multipliers;
      std::vector<typename F::optimization_vector*>
        sub_ov_ptrs(subfactor_ptrs_.size(), NULL);
      for (size_t i(0); i < subfactor_ptrs_.size(); ++i) {
        subfactor_ptrs_[i] = new F(*(other.subfactor_ptrs_[i]));
        sub_ov_ptrs[i] = &(subfactor_ptrs_[i]->weights());
      }
      ov = optimization_vector(sub_ov_ptrs);
      fixed_records_ = other.fixed_records_;
      x2_fixed_indices = other.x2_fixed_indices;
      return *this;
    }

    ~hybrid_crf_factor() {
      foreach(F* subfptr, subfactor_ptrs_)
        delete subfptr;
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
      out << "]\n";
      if (print_vals) {
        out << "x2_vec: " << x2_vec << "\n"
            << "Subfactors:\n";
        foreach(const finite_assignment& fa, assignments(make_domain(x2_vec))) {
          out << "For [" << fa << "], subfactor: " << subfactor(fa) << "\n";
        }
      }
      out << std::flush;
    }

    // Public methods: Probabilistic queries
    // =========================================================================

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const assignment_type& a) const {
      return subfactor(a).v(a);
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const record_type& r) const {
      return subfactor(r).v(r);
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const assignment_type& a) const {
      return subfactor(a).logv(a);
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const record_type& r) const {
      return subfactor(r).logv(r);
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
      return subfactor(a).condition(a);
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
      return subfactor(r).condition(r);
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
      if (subfactor_ptrs_.size() == 0)
        return true;
      return subfactor_ptrs_[0]->log_space();
    }

    //! Tries to change this factor's internal representation to log-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already log-space
    bool convert_to_log_space() {
      if (subfactor_ptrs_.size() == 0)
        return true;
      bool b(subfactor(0).convert_to_log_space());
      for (size_t i(1); i < subfactor_ptrs_.size(); ++i) {
        if (subfactor(i).convert_to_log_space() != b)
          assert(false);
      }
      return b;
    }

    //! Tries to change this factor's internal representation to real-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already real-space
    bool convert_to_real_space() {
      if (subfactor_ptrs_.size() == 0)
        return true;
      bool b(subfactor(0).convert_to_real_space());
      for (size_t i(1); i < subfactor_ptrs_.size(); ++i) {
        if (subfactor(i).convert_to_real_space() != b)
          assert(false);
      }
      return b;
    }

    /**
     * When called, this fixes this factor to use records of this type very
     * efficiently (setting fixed_records_ = true).
     * This option MUST be turned off before using this factor with records
     * with different variable orderings!
     */
    void fix_records(const record_type& r) {
      x2_fixed_indices.resize(x2_vec.size());
      for (size_t i(0); i < x2_vec.size(); ++i)
        x2_fixed_indices[i] = safe_get(*(r.finite_numbering_ptr), x2_vec[i]);
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
    void add_gradient(optimization_vector& grad, const record_type& r,
                      double w) const {
      size_t i(my_subfactor_index(r));
      subfactor(i).add_gradient(grad.subvector(i), r, w);
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
      size_t i(my_subfactor_index(r));
      subfactor(i).add_expected_gradient(grad.subvector(i), r, fy, w);
    }

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void
    add_combined_gradient(optimization_vector& grad, const record_type& r,
                          const output_factor_type& fy, double w = 1.) const {
      size_t i(my_subfactor_index(r));
      subfactor(i).add_combined_gradient(grad.subvector(i), r, fy, w);
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
      size_t i(my_subfactor_index(r));
      subfactor(i).add_hessian_diag(hessian.subvector(i), r, w);
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
      size_t i(my_subfactor_index(r));
      subfactor(i).add_expected_hessian_diag(hessian.subvector(i), r, fy, w);
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
      size_t i(my_subfactor_index(r));
      subfactor(i).add_expected_squared_gradient(sqrgrad.subvector(i), r, fy,w);
    }

    /**
     * Returns the regularization penalty for the current weights and
     * the given regularization parameters.
     */
    double regularization_penalty(const regularization_type& reg) const {
      assert(false); // This should depend on the values of X2.
      double val(0);
      foreach(const F* subfptr, subfactor_ptrs_)
        val += subfptr->regularization_penalty(reg);
      return val;
    }

    /**
     * Adds the gradient of the regularization term for the current weights
     * and the given regularization parameters.
     */
    void add_regularization_gradient(optimization_vector& grad,
                                     const regularization_type& reg,
                                     double w) const {
      assert(false); // This should depend on the values of X2.
      foreach(const finite_assignment& fa, assignments(make_domain(x2_vec))) {
        size_t i(my_subfactor_index(fa));
        subfactor(fa).add_regularization_gradient(grad.subvector(i), reg, w);
      }
    }

    /**
     * Adds the diagonal of the Hessian of the regularization term for the
     * current weights and the given regularization parameters.
     */
    void add_regularization_hessian_diag(optimization_vector& hd,
                                         const regularization_type& reg,
                                         double w) const {
      assert(false); // This should depend on the values of X2.
      foreach(const finite_assignment& fa, assignments(make_domain(x2_vec))) {
        size_t i(my_subfactor_index(fa));
        subfactor(fa).add_regularization_hessian_diag(hd.subvector(i), reg, w);
      }
    }

    // Methods used by learn_crf_factor
    // =========================================================================

    //! Given the variables in x2 in order, return the multipliers for
    //! computing subfactor indices.
    static std::vector<size_t>
    compute_multipliers(const finite_var_vector& x2_vec) {
      std::vector<size_t> x2_multipliers(x2_vec.size(), 0);
      if (x2_vec.size() > 0) {
        x2_multipliers[0] = 1;
        for (size_t i(1); i < x2_vec.size(); ++i)
          x2_multipliers[i] = x2_vec[i-1]->size() * x2_multipliers[i-1];
      }
      return x2_multipliers;
    }

    //! Given an assignment over x2, return the corresponding subfactor's index.
    static inline size_t
    subfactor_index(const finite_assignment& a,
                    const finite_var_vector& x2_vec,
                    const std::vector<size_t>& x2_multipliers) {
      size_t i(0);
      for (size_t j(0); j < x2_vec.size(); ++j)
        i += safe_get(a, x2_vec[j]) * x2_multipliers[j];
      return i;
    }

    //! Given a record over x2, return the corresponding subfactor's index.
    static inline size_t
    subfactor_index(const finite_record& r,
                    const finite_var_vector& x2_vec,
                    const std::vector<size_t>& x2_multipliers) {
      size_t i(0);
      for (size_t j(0); j < x2_vec.size(); ++j)
        i += r.finite(x2_vec[j]) * x2_multipliers[j];
      return i;
    }

    // Private data
    // =========================================================================
  private:

    //! This class owns this data.
    std::vector<F*> subfactor_ptrs_;

    //! Vector of variables in x2
    finite_var_vector x2_vec;

    //! Multipliers corresponding to variables in x2_vec.
    std::vector<size_t> x2_multipliers;

    optimization_vector ov;

    //! True if fix_records() has been called.
    bool fixed_records_;

    //! If fixed_records_ == true, then these indices correspond to the
    //! indices of the variables in x2_vec in the fixed record form.
    std::vector<size_t> x2_fixed_indices;

    // Private methods
    // =========================================================================

    //! Given an assignment over x2, return the corresponding subfactor's index.
    inline size_t my_subfactor_index(const finite_assignment& a) const {
      size_t i(subfactor_index(a, x2_vec, x2_multipliers));
      assert(i < subfactor_ptrs_.size());
      return i;
    }

    //! Given a record over x2, return the corresponding subfactor's index.
    inline size_t my_subfactor_index(const finite_record& r) const {
      size_t i(0);
      if (fixed_records_) {
        for (size_t j(0); j < x2_vec.size(); ++j)
          i += r.finite(x2_fixed_indices[j]) * x2_multipliers[j];
      } else {
        i = subfactor_index(r, x2_vec, x2_multipliers);
      }
      assert(i < subfactor_ptrs_.size());
      return i;
    }

    const F& subfactor(size_t i) const { return *(subfactor_ptrs_[i]); }

    F& subfactor(size_t i) { return *(subfactor_ptrs_[i]); }

    //! Given an assignment over x2, return the corresponding subfactor.
    const F& subfactor(const finite_assignment& a) const {
      return subfactor(my_subfactor_index(a));
    }

    //! Given a record over x2, return the corresponding subfactor.
    const F& subfactor(const finite_record& r) const {
      return subfactor(my_subfactor_index(r));
    }

    // Set ov using the current subfactor_ptrs_.
    void set_ov() {
      std::vector<typename F::optimization_vector*>
        sub_ov_ptrs(subfactor_ptrs_.size(), NULL);
      for (size_t i(0); i < subfactor_ptrs_.size(); ++i) {
        assert(subfactor_ptrs_[i]);
        sub_ov_ptrs[i] = &(subfactor_ptrs_[i]->weights());
      }
      ov = optimization_vector(sub_ov_ptrs);
    }

  };  // class hybrid_crf_factor

};  // namespace prl

#include <prl/macros_undef.hpp>

#endif // PRL_HYBRID_CRF_FACTOR_HPP
