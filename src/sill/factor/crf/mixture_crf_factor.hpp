#ifndef SILL_MIXTURE_CRF_FACTOR_HPP
#define SILL_MIXTURE_CRF_FACTOR_HPP

#include <sill/factor/crf/learnable_crf_factor.hpp>
#include <sill/factor/mixture.hpp>
#include <sill/optimization/hybrid_opt_vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * CRF factor which represents a mixture of other CRF factors.
   * Each element in the mixture P(Y|X) has the same Y,X variables.
   *
   * @tparam F  Type of CRF factor used for each mixture component.
   */
  template <typename F>
  class mixture_crf_factor
    : public learnable_crf_factor<typename F::input_variable_type,
                                  mixture<typename F::output_factor_type>,
                                  hybrid_opt_vector
                                  <typename F::optimization_vector>,
                                  F::regularization_type::nlambdas> {

    // Public types
    // =========================================================================
  public:

    //! Base class
    typedef learnable_crf_factor<typename F::input_variable_type,
                                 mixture<typename F::output_factor_type>,
                                 hybrid_opt_vector
                                 <typename F::optimization_vector>,
                                 F::regularization_type::nlambdas> base;

    //! Type of CRF factor used for each mixture component.
    typedef F subfactor_type;

    typedef typename base::input_variable_type input_variable_type;
    typedef typename base::input_domain_type input_domain_type;
    typedef typename base::input_assignment_type input_assignment_type;
    typedef typename base::input_record_type input_record_type;

    typedef typename base::assignment_type assignment_type;
    typedef typename base::record_type record_type;

    typedef typename base::output_domain_type output_domain_type;
    typedef typename base::output_factor_type output_factor_type;

    typedef typename base::domain_type domain_type;

    typedef typename base::optimization_vector optimization_vector;
    typedef typename base::regularization_type regularization_type;

    typedef typename subfactor_type::la_type la_type;

    //! Parameters used for learn_crf_factor.
    struct parameters {
    }; // struct parameters

    // Public methods: Constructors and Serialization
    // =========================================================================

    //! Default constructor.
    mixture_crf_factor() { }

    //! Constructor.
    mixture_crf_factor(const std::vector<subfactor_type>& comps)
      : base((comps.size() == 0
              ? output_domain_type()
              : comps[0].output_arguments()),
             (comps.size() == 0
              ? copy_ptr<input_domain_type>(new input_domain_type())
              : comps[0].input_arguments_ptr())),
        components_(comps) {
      for (size_t i = 1; i < comps.size(); ++i) {
        if (comps[i].output_arguments() != output_arguments() ||
            comps[i].input_arguments() != input_arguments()) {
          throw std::invalid_argument
            ("mixture_crf_factor constructor given bad set of components!");
        }
      }
    }

    //! Constructor: conversion from a mixture.
    mixture_crf_factor(const mixture<output_factor_type>& mix,
                       const output_domain_type& Y,
                       const input_domain_type& X)
      : base(Y, X) {
      domain_type YX(set_union(Y,X));
      foreach(const output_factor_type& f, mix.components()) {
        if (f.arguments() != YX) {
          throw std::invalid_argument
            ("mixture_crf_factor constructor given bad mix,Y,X!");
        }
        components_.push_back(subfactor_type(f, Y, X));
      }
    }

    //! Serialize members
    void save(oarchive & ar) const {
      base::save(ar);
      ar << components_;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      base::load(ar);
      ar >> components_;
    }

    void relabel_outputs_inputs(const output_domain_type& new_Y,
                                const input_domain_type& new_X) {
      assert(false); // TO DO
    }

    // Public methods: Getters
    // =========================================================================

    using base::output_arguments;
    using base::input_arguments;
    using base::input_arguments_ptr;

    //! Vector of components in this mixture.
    const std::vector<subfactor_type>& components() const {
      return components_;
    }

    // Public methods: Probabilistic queries
    // =========================================================================

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const assignment_type& a) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in real-space (not log-space).
    double v(const record_type& r) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const assignment_type& a) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    //! Evaluates this factor for the given datapoint, returning its value
    //! in log-space.
    double logv(const record_type& r) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param a  This must assign values to all X in this factor
     *           (but may assign values to any other variables as well).
     * @return  factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const output_factor_type&
    condition(const input_assignment_type& a) const {
      conditioned_f = output_factor_type(components_.size());
      for (size_t i = 0; i < components_.size(); ++i) {
        conditioned_f[i] =
          typename F::output_factor_type(components_[i].condition(a));
      }
      return conditioned_f;
    }

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param r Record with values for X in this factor
     *          (which may have values for any other variables as well).
     * @return  factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    const output_factor_type&
    condition(const input_record_type& r) const {
      conditioned_f = output_factor_type(components_.size());
      for (size_t i = 0; i < components_.size(); ++i) {
        conditioned_f[i] =
          typename F::output_factor_type(components_[i].condition(r));
      }
      return conditioned_f;
    }

    // Public methods: Learning-related methods
    // =========================================================================

    //! @return  true iff the data is stored in log-space
    bool log_space() const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    //! Tries to change this factor's internal representation to log-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already log-space
    bool convert_to_log_space() {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    //! Tries to change this factor's internal representation to real-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already real-space
    bool convert_to_real_space() {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    const optimization_vector& weights() const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    optimization_vector& weights() {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    // Public methods: Learning methods from learnable_crf_factor
    // =========================================================================

    //! Adds the gradient of the log of this factor w.r.t. the weights,
    //! evaluated at the given datapoint with the current weights.
    //! @param grad   Pre-allocated vector to which to add the gradient.
    //! @param r      Datapoint.
    //! @param w      Weight by which to multiply the added values.
    void
    add_gradient(optimization_vector& grad, const record_type& r,
                 double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
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
    void
    add_expected_gradient(optimization_vector& grad, const input_record_type& r,
                          const output_factor_type& fy, double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    /**
     * This is equivalent to (but faster than) calling:
     *   add_gradient(grad, r, w);
     *   add_expected_gradient(grad, r, fy, -1 * w);
     */
    void
    add_combined_gradient(optimization_vector& grad, const record_type& r,
                          const output_factor_type& fy, double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
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
                     double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
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
                              const output_factor_type& fy,
                              double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
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
    add_expected_squared_gradient(optimization_vector& sqrgrad,
                                  const input_record_type& r,
                                  const output_factor_type& fy,
                                  double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    /**
     * Returns the regularization penalty for the current weights and
     * the given regularization parameters.
     * This is:  - .5 * lambda * inner_product(weights, weights)
     */
    double
    regularization_penalty(const regularization_type& reg) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    /**
     * Adds the gradient of the regularization term for the current weights
     * and the given regularization parameters.
     * This is:  - lambda * weights
     */
    void
    add_regularization_gradient(optimization_vector& grad,
                                const regularization_type& reg,
                                double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    /**
     * Adds the diagonal of the Hessian of the regularization term for the
     * current weights and the given regularization parameters.
     */
    void
    add_regularization_hessian_diag(optimization_vector& hd,
                                    const regularization_type& reg,
                                    double w = 1) const {
      throw std::runtime_error
        ("mixture_crf_factor HAS LOTS OF UNIMPLEMENTED METHODS!");
    }

    // Protected data
    // =========================================================================
  protected:

    using base::Ydomain_;
    using base::Xdomain_ptr_;

    std::vector<subfactor_type> components_;

    mutable output_factor_type conditioned_f;

  }; // class mixture_crf_factor

};  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_MIXTURE_CRF_FACTOR_HPP
