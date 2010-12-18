#ifndef SILL_FACTOR_CONCEPTS_HPP
#define SILL_FACTOR_CONCEPTS_HPP

#include <iostream>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include <sill/global.hpp>
#include <sill/learning/crossval_parameters.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/range/concepts.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //============================================================================
  // MARGINAL FACTORS
  //============================================================================

  /**
   * A concept that represents a factor.
   *
   * A factor needs to be default constructible, copy constructible,
   * assignable, and implement a unary function from its domain to the
   * range. A factor needs to provide a number of elementary operations:
   * a non-modifying binary combine operation, unary restrict and collapse
   * operations, as well as modifying normalize and variable substitution
   * operations.
   *
   * The library defines a number of convenience free (non-member)
   * functions that call the basic factor operations described above.
   * These functions are automatically included in the factor.hpp header.
   *
   * Typically, factors define constructors and operators that convert
   * from / to other types. For example, a table_factor defines a
   * conversion constructor from a constant_factor.  It also defines a
   * conversion operator to constant_factor (this latter conversion
   * fails if the table_factor has non-empty argument set). By
   * convention, the more complex factors define the conversions
   * from/to the more primitive factors. Since the conversion from a
   * primitive datatype is always valid, it is not declared as
   * explicit. The conversion to a primitive datatype may not be
   * valid, as indicated above in the case of converting a
   * table_factor to a constant_factor.  Such conversions are always
   * checked at runtime.
   *
   * \ingroup factor_concepts
   * \see constant_factor, table_factor
   */
  template <typename F>
  struct Factor
    : DefaultConstructible<F>, CopyConstructible<F>, Assignable<F> {

    /**
     * The type that represents the value returned by factor's
     * operator() and norm_constant().  Typically, this type is either
     * double or logarithmic<double>.
     */
    typedef typename F::result_type result_type;

    /**
     * The type of variables, used by the factor. Typically, this type is
     * either sill::variable or its descendant.
     */
    typedef typename F::variable_type variable_type;
    
    /**
     * The type that represents the factor's domain, that is, the set of
     * factor's arguments. This type must be equal to set<variable_type*>.
     */
    typedef typename F::domain_type domain_type;
    
    
    /**
     * The type that represents an assignment. 
     * In the current design, this typedef is not required to be a part of the
     * factor's public interface, but the factor must use this type in its
     * operator(). 
     */
    typedef typename F::assignment_type assignment_type;

    
    /**
     * The supported combine operations.
     * Similarly, this flag represents the supported combine operations
     * between F and other factor types.
     */
//    static const unsigned combine_ops = F::combine_ops;

    //! Returns the arguments of the factor
    const domain_type& arguments() const;

    //! Combines the given factor into this factor with a binary operation
    F& combine_in(const F& f, op_type op);

    //! Returns a new factor which represents the result of restricting this
    //! factor to an assignment to one or more of its variables
    F restrict(const assignment_type& a) const;

    /**
     * Renames the arguments of this factor in-place.
     *
     * @param map
     *        an object such that map[v] maps the variable handle v
     *        a type compatible variable handle; this mapping must be 1:1.
     *
     * \todo Requires that the keys and values of var_map are disjoint
     */
    F& subst_args(const std::map<variable_type*, variable_type*>& map);

    /* Evaluates the factor for the given assignment. */
    result_type operator()(const assignment_type& a);

    concept_usage(Factor) {
      F f;
      const F& cf = f;
      std::map<variable_type*, variable_type*> vm;
      assignment_type a;
      domain_type d;

      // member functions
      sill::same_type(cf.arguments(), d);
      sill::same_type(f.combine_in(f, product_op), f);
      sill::same_type(f.subst_args(vm), f);

      // static functions
      sill::same_type(combine(cf, cf, product_op), f);
      f.restrict(a);
    }

  }; // concept Factor

  /**
   * A concept that represents a distribution and a factor. 
   *
   * \ingroup factor_concepts
   * \see table_factor
   */
  template <typename F>
  struct DistributionFactor : Factor<F> {

    //! Multiplies this factor by another factor
    F& operator*=(const F& f);

    //! Divides this factor by another factor
    F& operator/=(const F& f);

    //! Computes a marginal (sum) over a factor expression
    F marginal(const typename F::domain_type& retain) const;

    //! Returns true if the factor is normalizable (default implementation)
    bool is_normalizable() const;

    //! Returns the normalization constant of this factor
    typename F::result_type norm_constant() const;

    /**
     * Normalizes a factor in-place.  This method throws an exception if
     * the supplied factor is not normalizable (because its integral is
     * not positive and finite).
     */
    F& normalize();

    //! Computes the entropy of the distribution.
    double entropy() const;

    //! Computes KL(*this, q)
    double relative_entropy(const DistributionFactor& q);

    concept_usage(DistributionFactor) {
      bool b;
      typename F::result_type r;
      F f;
      const F& cf = f;

      // factor operations
//      sill::same_type(f, f*=f); // TO DO: THESE ARE DISABLED B/C THESE OPERATIONS NEED TO BE PART OF THE FACTOR INTERFACE, NOT FREE FUNCTIONS; GET RID OF THE FREE FUNCTIONS IN operations.hpp
//      sill::same_type(f, f/=f);
      sill::same_type(f, f.marginal(typename F::domain_type()));
      sill::same_type(b, f.is_normalizable());
      sill::same_type(f.normalize(), f);
      r = f.norm_constant();

      // entropy computations
      cf.entropy();
      cf.relative_entropy(cf);
    }

  }; // concept DistributionFactor

  /**
   * A concept that represents a distribution and a factor, which has additional
   * methods which help with learning.
   *
   * \ingroup factor_concepts
   * \see table_factor
   */
  template <typename F>
  struct LearnableDistributionFactor : DistributionFactor<F> {

    typedef typename DistributionFactor<F>::assignment_type assignment_type;
    typedef typename DistributionFactor<F>::domain_type domain_type;

    /**
     * Learns a marginal factor P(X) from data.
     * @param X          Variables in marginal.
     * @param ds         Training data.
     * @param smoothing  Regularization (>= 0).
     */
    static F
    learn_marginal(const typename F::domain_type& X, const dataset& ds,
                   double smoothing);

    /**
     * Learns a conditional factor P(A | B, C=c) from data.
     * @param ds         Training data.
     * @param smoothing  Regularization (>= 0).
     */
    static F
    learn_conditional(const typename F::domain_type& A,
                      const typename F::domain_type& B,
                      const typename F::assignment_type& c, const dataset& ds,
                      double smoothing);

    concept_usage(LearnableDistributionFactor) {
      sill::same_type(f, F::learn_marginal(dom_constref, ds_constref, d));
      sill::same_type(f, F::learn_conditional(dom_constref, dom_constref,
                                             a_constref, ds_constref, d));
    }

  private:
    F f;
    static const domain_type& dom_constref;
    static const dataset& ds_constref;
    double d;
    static const assignment_type& a_constref;

  }; // concept LearnableDistributionFactor

  //============================================================================
  // CONDITIONAL FACTORS
  //============================================================================

  /**
   * The concept that represents a CRF factor/potential.
   *
   * A CRF factor is an arbitrary function Phi(Y,X) which is part of a CRF model
   * P(Y | X) = (1/Z(X)) \prod_i Phi_i(Y_{C_i},X_{C_i}).
   * This allows support of a variety of factors, such as:
   *  - a table_factor over finite variables in X,Y
   *  - a logistic regression function which supports vector variables in X
   *  - Gaussian factors
   *
   * CRF factors can be arbitrary functions, but it is generally easier to think
   * of a CRF factor as an exponentiated sum of weights times feature values:
   *  Phi(Y,X) = \exp[ \sum_j w_j * f_j(Y,X) ]
   * where w_j are fixed (or learned) weights and f_j are arbitrary functions.
   * Since CRF parameter learning often requires the parameters to be in
   * log-space but inference often requires the parameters to be in real-space,
   * CRF factors support both:
   *  - They maintain a bit indicating whether their data is stored
   *    in log-space.
   *  - They have a method which tries to change the internal format
   *    between log- and real-space.
   *  - The learning methods explicitly state what space they return values in.
   * These concepts from parameter learning are in CRFfactor since it makes
   * things more convenient for crf_model (instead of keeping the learning
   * concepts within the LearnableCRFfactor concept class).
   *
   * @author Joseph Bradley
   */
  template <class F>
  struct CRFfactor
    : DefaultConstructible<F>, CopyConstructible<F>, Assignable<F> {

    // Public types
    // =========================================================================

    /**
     * The type that represents the value returned by factor's
     * operator() and norm_constant().  Typically, this type is either
     * double or logarithmic<double>.
     */
    typedef typename F::result_type result_type;

    /**
     * The type of input variables used by the factor.
     * Typically, this type is either sill::variable or its descendant.
     */
    typedef typename F::input_variable_type input_variable_type;

    /**
     * The type of output variables used by the factor.
     * Typically, this type is either sill::variable or its descendant.
     */
    typedef typename F::output_variable_type output_variable_type;

    /**
     * The type of variables used by the factor.
     * Typically, this type is either sill::variable or its descendant.
     * Both input_variable_type and output_variable_type should inherit
     * from this type.
     */
    typedef typename F::variable_type variable_type;

    /**
     * The type that represents the factor's input variable domain,
     * that is, the set of input arguments X in the factor.
     * This type must be equal to set<input_variable_type*>.
     */
    typedef typename F::input_domain_type input_domain_type;

    /**
     * The type that represents the factor's output variable domain,
     * that is, the set of output arguments Y in the factor.
     * This type must be equal to set<output_variable_type*>.
     */
    typedef typename F::output_domain_type output_domain_type;

    /**
     * The type that represents the factor's variable domain,
     * that is, the set of arguments X,Y in the factor.
     * This type must be equal to set<variable_type*>.
     */
    typedef typename F::domain_type domain_type;

    /**
     * The type that represents an assignment to input variables.
     */
    typedef typename F::input_assignment_type input_assignment_type;

    /**
     * The type that represents an assignment to output variables.
     */
    typedef typename F::output_assignment_type output_assignment_type;

    /**
     * The type that represents an assignment.
     */
    typedef typename F::assignment_type assignment_type;

    /**
     * The type which this factor f(Y,X) outputs to represent f(Y, X=x).
     * For finite Y, this will probably be table_factor;
     * for vector Y, this will probably be a subtype of gaussian_factor.
     */
    typedef typename F::output_factor_type output_factor_type;

    //! Type which parametrizes this factor, usable for optimization and
    //! learning.
    typedef typename F::optimization_vector optimization_vector;

    // Public methods: Constructors, getters, helpers
    // =========================================================================

    //! @return  output variables in Y for this factor
    const output_domain_type& output_arguments() const;

    //! @return  input variables in X for this factor
    const input_domain_type& input_arguments() const;

    //! @return  input variables in X for this factor
    copy_ptr<input_domain_type> input_arguments_ptr() const;

    //! It may be faster to use input_arguments(), output_arguments().
    //! @return  variables in Y,X for this factor
    domain_type arguments() const;

    //! @return  true iff the data is stored in log-space
    bool log_space() const;

    //! Tries to change this factor's internal representation to log-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already log-space
    bool convert_to_log_space();

    //! Tries to change this factor's internal representation to real-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already real-space
    bool convert_to_real_space();

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    const optimization_vector& weights() const;

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    optimization_vector& weights();

    //! If true, then this is not a learnable factor.
    //! (I.e., the factor's value will be fixed during learning.)
    //! (default after construction = false)
    bool fixed_value() const;

    //! If true, then this is not a learnable factor.
    //! (I.e., the factor's value will be fixed during learning.)
    //! This returns a mutable reference.
    //! (default after construction = false)
    bool& fixed_value();

    void print(std::ostream& out) const;

    // Public methods: Probabilistic queries
    // =========================================================================

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param a  This must assign values to all X in this factor
     *           (but may assign values to any other variables as well).
     * @return  table factor representing the factor with
     *          the given input variable (X) instantiation
     */
    const output_factor_type& condition(const assignment& a) const;

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param r Record with values for X in this factor
     *          (which may have values for any other variables as well).
     * @return  table factor representing the factor with
     *          the given input variable (X) instantiation
     */
    const output_factor_type& condition(const record& r) const;

    /**
     * Returns the empirical expectation of the log of this factor.
     * In particular, if this factor represents P(A|B), then
     * this returns the expected log likelihood of the distribution P(A | B).
     */
    double log_expected_value(const dataset& ds) const;

    concept_usage(CRFfactor) {
      sill::same_type(f_const_ref.output_arguments(), od_const_ref);
      sill::same_type(f_const_ref.input_arguments(), id_const_ref);
      sill::same_type(f_const_ref.input_arguments_ptr(), id_copy_ptr);
      sill::same_type(f_const_ref.arguments(), dom);
      sill::same_type(f_const_ref.log_space(), b);
      sill::same_type(f_ptr->convert_to_log_space(), b);
      sill::same_type(f_ptr->convert_to_real_space(), b);
      sill::same_type(f_const_ref.weights(), opt_vec_const_ref);
      sill::same_type(f_ptr->weights(), opt_vec_ref);
      sill::same_type(f_const_ref.fixed_value(), b);
      sill::same_type(f_ref.fixed_value(), b_ref);
      f_const_ref.print(out);

      sill::same_type(f_const_ref.condition(a_const_ref), of_const_ref);
      sill::same_type(f_const_ref.condition(rec_const_ref), of_const_ref);
      sill::same_type(f_const_ref.log_expected_value(ds_const_ref), d);
      out << f_const_ref;
    }

  private:
    static F& f_ref;
    static const F& f_const_ref;
    static const output_domain_type& od_const_ref;
    static const input_domain_type& id_const_ref;
    copy_ptr<input_domain_type> id_copy_ptr;
    domain_type dom;
    bool b;
    static bool& b_ref;
    F* f_ptr;
    static const optimization_vector& opt_vec_const_ref;
    static optimization_vector& opt_vec_ref;
    static std::ostream& out;

    static const record& rec_const_ref;
    static const output_factor_type& of_const_ref;

    static const assignment& a_const_ref;
    static const dataset& ds_const_ref;
    double d;

  };  // struct CRFfactor

  /**
   * CRFfactor which supports gradient-based parameter learning,
   * as well as structure learning using pwl_crf_learner.
   */
  template <class F>
  struct LearnableCRFfactor
    : public CRFfactor<F> {

    // Public types
    // =========================================================================

    // Import types from base class
    typedef typename CRFfactor<F>::result_type result_type;
    typedef typename CRFfactor<F>::input_variable_type input_variable_type;
    typedef typename CRFfactor<F>::output_variable_type output_variable_type;
    typedef typename CRFfactor<F>::variable_type variable_type;
    typedef typename CRFfactor<F>::input_domain_type input_domain_type;
    typedef typename CRFfactor<F>::output_domain_type output_domain_type;
    typedef typename CRFfactor<F>::domain_type domain_type;
    typedef typename CRFfactor<F>::input_assignment_type input_assignment_type;
    typedef typename CRFfactor<F>::output_assignment_type
      output_assignment_type;
    typedef typename CRFfactor<F>::assignment_type assignment_type;
    typedef typename CRFfactor<F>::output_factor_type output_factor_type;
    typedef typename CRFfactor<F>::optimization_vector optimization_vector;

    /**
     * Type of parameters passed to learning methods.
     * This should have at least this value:
     *  - regularization_type reg
     */
    typedef typename F::parameters parameters;

    /**
     * Regularization information.  This should contain, e.g., the type
     * of regularization being used and the strength.
     * This should have 3 values:
     *  - size_t regularization: type of regularization
     *  - static const size_t nlambdas: dimensionality of lambdas
     *  - vec lambdas: regularization parameters
     */
    typedef typename F::regularization_type regularization_type;

    // Public methods: Learning methods
    // =========================================================================

    /**
     * Adds the gradient of the log of this factor w.r.t. the weights,
     * evaluated at the given datapoint with the current weights.
     * @param grad   Pre-allocated vector to which to add the gradient.
     * @param r      Datapoint.
     * @param w      Weight by which to multiply the added values.
     */
    void
    add_gradient(optimization_vector& grad, const record& r, double w) const;

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
    void add_expected_gradient(optimization_vector& grad, const record& r,
                               const output_factor_type& fy, double w) const;

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void add_combined_gradient(optimization_vector& grad, const record& r,
                               const output_factor_type& fy, double w) const;

    /**
     * Adds the diagonal of the Hessian of the log of this factor w.r.t. the
     * weights, evaluated at the given datapoint with the current weights.
     * @param hessian Pre-allocated vector to which to add the hessian.
     * @param r       Datapoint.
     * @param w       Weight by which to multiply the added values.
     */
    void
    add_hessian_diag(optimization_vector& hessian, const record& r,
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
    template <typename YFactor>
    void
    add_expected_hessian_diag(optimization_vector& hessian, const record& r,
                              const YFactor& fy, double w) const;

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
    template <typename YFactor>
    void
    add_expected_squared_gradient(optimization_vector& sqrgrad, const record& r,
                                  const YFactor& fy, double w) const;

    /**
     * Returns the regularization penalty for the current weights and
     * the given regularization parameters.
     */
    double regularization_penalty(const regularization_type& reg) const;

    /**
     * Adds the gradient of the regularization term for the current weights
     * and the given regularization parameters.
     * @param w       Weight by which to multiply the added values.
     */
    void add_regularization_gradient(optimization_vector& grad,
                                     const regularization_type& reg,
                                     double w) const;

    /**
     * Adds the diagonal of the Hessian of the regularization term for the
     * current weights and the given regularization parameters.
     * @param w       Weight by which to multiply the added values.
     */
    void add_regularization_hessian_diag(optimization_vector& hd,
                                         const regularization_type& reg,
                                         double w) const;

    /**
     * Returns a newly allocated factor which represents P(Y | X),
     * learned from data.
     */
    /*
    static F*
    learn_crf_factor
    (boost::shared_ptr<dataset> ds_ptr,
     const output_domain_type& Y_, copy_ptr<input_domain_type> X_ptr_,
     unsigned random_seed, const parameters& params);
    */
    /**
     * Returns a newly allocated factor which represents P(Y | X),
     * learned from data.
     * This does cross validation to choose regularization parameters.
     *
     * @param reg_params (Return value.) Parameters which were tried.
     * @param means      (Return value.) Means of scores for the given lambdas.
     * @param stderrs    (Return value.) Std errors of scores for the lambdas.
     * @param cv_params  Parameters specifying how to do cross validation.
     */
    /*
    static F*
    learn_crf_factor_cv
    (std::vector<regularization_type>& reg_params, vec& means, vec& stderrs,
     const crossval_parameters<regularization_type::nlambdas>& cv_params,
     boost::shared_ptr<dataset> ds_ptr,
     const output_domain_type& Y_, copy_ptr<input_domain_type> X_ptr_,
     unsigned random_seed, const parameters& params);
    */

    concept_usage(LearnableCRFfactor) {
      sill::same_type(params_const_ref.valid(), b);
      sill::same_type(params_const_ref.reg, reg);

      sill::same_type(reg.regularization, tmpsize);
      sill::same_type(reg.nlambdas, tmpsize);
      sill::same_type(reg.lambdas, tmpvec);

      f_const_ref.add_gradient(opt_vec_ref, rec_const_ref, d);
      f_const_ref.add_expected_gradient(opt_vec_ref, rec_const_ref,
                                        of_const_ref, d);
//      f_const_ref.add_combined_gradient(opt_vec_ref, rec_const_ref,
//                                        of_const_ref, d);
      f_const_ref.add_hessian_diag(opt_vec_ref, rec_const_ref, d);
      f_const_ref.add_expected_hessian_diag(opt_vec_ref, rec_const_ref,
                                            of_const_ref, d);
      f_const_ref.add_expected_squared_gradient(opt_vec_ref, rec_const_ref,
                                                of_const_ref, d);

      sill::same_type(d, f_const_ref.regularization_penalty(reg));
      f_const_ref.add_regularization_gradient(opt_vec_ref, reg, d);
      f_const_ref.add_regularization_hessian_diag(opt_vec_ref, reg, d);
      /*
      sill::same_type(learn_crf_factor<F>(ds_shared_ptr, od_const_ref,
                                        id_copy_ptr, uns, params_const_ref),
                     f_ptr);
      sill::same_type(learn_crf_factor_cv<F>
                     (reg_params, vec_ref, vec_ref, cv_params_const_ref,
                      ds_shared_ptr, od_const_ref, id_copy_ptr, uns,
                      params_const_ref),
                     f_ptr);
      */
    }

  private:
    static const F& f_const_ref;
    static const output_domain_type& od_const_ref;
    static const input_domain_type& id_const_ref;
    copy_ptr<input_domain_type> id_copy_ptr;
    domain_type dom;
    bool b;
    F* f_ptr;
    static const optimization_vector& opt_vec_const_ref;
    static optimization_vector& opt_vec_ref;
    static std::ostream& out;
    static const record& rec_const_ref;
    static const output_factor_type& of_const_ref;
    static const assignment& a_const_ref;
    static const dataset& ds_const_ref;
    double d;

    static const parameters& params_const_ref;

    size_t tmpsize;
    vec tmpvec;

    static const regularization_type& reg;
    boost::shared_ptr<dataset> ds_shared_ptr;
    unsigned uns;

    output_factor_type of;

    static std::vector<regularization_type>& reg_params;
    static vec& vec_ref;
    static const vec& vec_const_ref;

    static const crossval_parameters<regularization_type::nlambdas>&
      cv_params_const_ref;

  }; // struct LearnableCRFfactor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

