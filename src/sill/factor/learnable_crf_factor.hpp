#ifndef SILL_LEARNABLE_CRF_FACTOR_HPP
#define SILL_LEARNABLE_CRF_FACTOR_HPP

#include <sill/factor/crf_factor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Regularization information for CRF parameters.
   * This contains the type of regularization being used and the strength.
   * Look in the subclasses (e.g., table_crf_factor, gaussian_crf_factor)
   * for more info on allowed values and their meanings.
   *
   * @tparam NLambdas     Dimensionality of the vector of regularization
   *                      parameters used in learning.
   */
  template <size_t NLambdas>
  struct crf_regularization_type {

    //! All CRF factors should support:
    //!  - 0: none
    //!  - 2: L2 regularization (default).
    size_t regularization;

    static const size_t nlambdas = NLambdas;

    //! Strength of regularization.
    //!  (default = 0)
    vec lambdas;

    crf_regularization_type()
      : regularization(2), lambdas(zeros<vec>(NLambdas)) { }

  }; // struct crf_regularization_type

  template <size_t NLambdas>
  std::ostream&
  operator<<(std::ostream& out, const crf_regularization_type<NLambdas>& reg) {
    switch (reg.regularization) {
    case 0:
      out << "No regularization";
      break;
    case 2:
      out << "L2 reg " << reg.lambdas;
      break;
    default:
      assert(false);
    }
    return out;
  }

  /**
   * A virtual base class for a CRF factor/potential which supports learning.
   *
   * See the crf_factor interface for more info about CRF factors.
   *
   * @tparam InputVar     Type of input variable.
   * @tparam OutputFactor Type of factor resulting from conditioning.
   * @tparam OptVector    Type used to represent the factor weights
   *                      (which must fit the OptimizationVector concept
   *                      for crf_parameter_learner).
   * @tparam NLambdas     Dimensionality of the vector of regularization
   *                      parameters used in learning.
   * @tparam LA           Linear algebra type specifier
   *                       (default = dense_linear_algebra<>)
   *
   * \ingroup factor
   * @author Joseph Bradley
   */
  template <typename InputVar, typename OutputFactor, typename OptVector,
            size_t NLambdas, typename LA = dense_linear_algebra<> >
  class learnable_crf_factor
    : public crf_factor<InputVar, OutputFactor, OptVector, LA> {

    // Public types
    // =========================================================================
  public:

    typedef LA la_type;

    //! Base class
    typedef crf_factor<InputVar, OutputFactor, OptVector, LA> base;

    // Import types from base.
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

    typedef crf_regularization_type<NLambdas> regularization_type;

    // Public methods: Constructors
    // =========================================================================
  public:

    //! Default constructor.
    learnable_crf_factor()
      : base() { }

    //! Constructor.
    learnable_crf_factor(const output_domain_type& Ydomain_,
                         copy_ptr<input_domain_type> Xdomain_ptr_)
      : base(Ydomain_, Xdomain_ptr_) { }

    //! Constructor.
    //! It is better to use the other constructor which takes a copy_ptr to
    //! the X domain (to avoid copies).
    learnable_crf_factor(const output_domain_type& Ydomain_,
                         const input_domain_type& Xdomain_)
      : base(Ydomain_,
             copy_ptr<input_domain_type>(new input_domain_type(Xdomain_))) { }

    virtual ~learnable_crf_factor() { }

    using base::save;
    using base::load;

    // Public methods: Learning methods
    // =========================================================================

    //! Adds the gradient of the log of this factor w.r.t. the weights,
    //! evaluated at the given datapoint with the current weights.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param w      Weight by which to multiply the added values.
    virtual void
    add_gradient(optimization_vector& grad, const record_type& r,
                 double w = 1) const = 0;

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
    virtual void
    add_expected_gradient(optimization_vector& grad, const input_record_type& r,
                          const output_factor_type& fy, double w = 1) const = 0;

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    virtual void
    add_combined_gradient(optimization_vector& grad, const record_type& r,
                          const output_factor_type& fy, double w = 1) const = 0;

    /**
     * Adds the diagonal of the Hessian of the log of this factor w.r.t. the
     * weights, evaluated at the given datapoint with the current weights.
     * @param hessian Pre-allocated vector to which to add the hessian.
     * @param r       Datapoint.
     * @param w       Weight by which to multiply the added values.
     */
    virtual void
    add_hessian_diag(optimization_vector& hessian, const record_type& r,
                     double w = 1) const = 0;

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
    virtual void
    add_expected_hessian_diag(optimization_vector& hessian,
                              const input_record_type& r,
                              const output_factor_type& fy,
                              double w = 1) const = 0;

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
    virtual void
    add_expected_squared_gradient(optimization_vector& sqrgrad,
                                  const input_record_type& r,
                                  const output_factor_type& fy,
                                  double w = 1) const = 0;

    /**
     * Returns the regularization penalty for the current weights and
     * the given regularization parameters.
     * This is:  - .5 * lambda * inner_product(weights, weights)
     */
    virtual double
    regularization_penalty(const regularization_type& reg) const = 0;

    /**
     * Adds the gradient of the regularization term for the current weights
     * and the given regularization parameters.
     * This is:  - lambda * weights
     */
    virtual void
    add_regularization_gradient(optimization_vector& grad,
                                const regularization_type& reg,
                                double w = 1) const = 0;

    /**
     * Adds the diagonal of the Hessian of the regularization term for the
     * current weights and the given regularization parameters.
     */
    virtual void
    add_regularization_hessian_diag(optimization_vector& hd,
                                    const regularization_type& reg,
                                    double w = 1) const = 0;

  }; // class learnable_crf_factor

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_LEARNABLE_CRF_FACTOR_HPP
