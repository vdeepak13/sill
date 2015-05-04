#ifndef SILL_CRF_FACTOR_HPP
#define SILL_CRF_FACTOR_HPP

#include <sill/base/variable_type_group.hpp>
#include <sill/base/variable_type_union.hpp>
#include <sill/learning/dataset_old/dataset.hpp>
#include <sill/model/model_functors.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A virtual base class for a CRF factor/potential.
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
   * @tparam InputVar     Type of input variable.
   * @tparam OutputFactor Type of factor resulting from conditioning.
   * @tparam OptVector    Type used to represent the factor weights
   *                       (which must fit the OptimizationVector concept
   *                       for crf_parameter_learner).
   * @tparam LA           Linear algebra type specifier
   *                       (default = dense_linear_algebra<>)
   *
   * @see learnable_crf_factor
   *
   * \ingroup factor
   * @author Joseph Bradley
   */
  template <typename InputVar, typename OutputFactor, typename OptVector,
            typename LA = dense_linear_algebra<> >
  class crf_factor {

    // Public types
    // =========================================================================
  public:

    typedef LA la_type;

    // Input type group
    //--------------------

    //! The type of input variables used by the factor.
    //! Typically, this type is either sill::variable or its descendant.
    typedef InputVar input_variable_type;

    typedef variable_type_group<input_variable_type,la_type> ivar_group;

    //! Type for the factor's input variable domain,
    //! that is, the set of input arguments X in the factor.
    typedef typename ivar_group::domain_type  input_domain_type;

    //! Vector of input variables.
    typedef typename ivar_group::var_vector_type  input_var_vector_type;

    //! Type for an assignment to input variables.
    typedef typename ivar_group::assignment_type  input_assignment_type;

    //! Type for a record for input variables.
    typedef typename ivar_group::record_type  input_record_type;

    // Output type group
    //--------------------

    //! The type of output variables used by the factor.
    //! Typically, this type is either sill::variable or its descendant.
    typedef typename OutputFactor::variable_type  output_variable_type;

    //! Type for the factor's output variable domain,
    //! that is, the set of output arguments Y in the factor.
    typedef typename OutputFactor::domain_type  output_domain_type;

    //! Vector of output variables.
    typedef typename OutputFactor::var_vector_type  output_var_vector_type;

    //! Type for an assignment to output variables.
    typedef typename OutputFactor::assignment_type  output_assignment_type;

    //! Type for a record for output variables.
    typedef typename OutputFactor::record_type  output_record_type;

    // Input + output type group
    //--------------------------

    /**
     * Both input_variable_type and output_variable_type inherit from this type.
     * Typically, this type is either sill::variable or its descendant.
     */
    typedef typename
    variable_type_union<input_variable_type,output_variable_type>::union_type
    variable_type;

    typedef variable_type_group<variable_type, la_type> var_group;

    /**
     * Type for the factor's variable domain,
     * that is, the set of arguments X,Y in the factor.
     */
    typedef typename var_group::domain_type  domain_type;

    //! Vector of variables.
    typedef typename var_group::var_vector_type  var_vector_type;

    //! Type for an assignment to X,Y.
    typedef typename var_group::assignment_type  assignment_type;

    //! Type for a record over X,Y.
    typedef typename var_group::record_type  record_type;

    // Other types
    //------------

    /**
     * Type for the value returned by factor's operator() and norm_constant().
     * Typically, this type is either double or logarithmic<double>.
     */
    typedef typename OutputFactor::result_type result_type;

    /**
     * The type which this factor f(Y,X) outputs to represent f(Y, X=x).
     * For finite Y, this will probably be table_factor;
     * for vector Y, this will probably be canonical_gaussian.
     */
    typedef OutputFactor output_factor_type;

    //! Type which parametrizes this factor, used in optimization and learning.
    typedef OptVector optimization_vector;

    // Public methods: Constructors
    // =========================================================================
  public:

    //! Default constructor.
    crf_factor()
      : Xdomain_ptr_(new input_domain_type()), fixed_value_(false) { }

    //! Constructor.
    crf_factor(const output_domain_type& Ydomain_,
               copy_ptr<input_domain_type> Xdomain_ptr_)
      : Ydomain_(Ydomain_), Xdomain_ptr_(Xdomain_ptr_), fixed_value_(false) { }

    virtual ~crf_factor() { }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << Ydomain_ << (*Xdomain_ptr_) << fixed_value_;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> Ydomain_ >> (*Xdomain_ptr_) >> fixed_value_;
    }

    // Public methods: Getters and helpers
    // =========================================================================

    //! @return  output variables in Y for this factor
    const output_domain_type& output_arguments() const {
      return Ydomain_;
    }

    //! @return  input variables in X for this factor
    const input_domain_type& input_arguments() const {
      return *Xdomain_ptr_;
    }

    //! @return  input variables in X for this factor
    copy_ptr<input_domain_type> input_arguments_ptr() const {
      return Xdomain_ptr_;
    }

    //! It is faster to use input_arguments(), output_arguments().
    //! @return  variables in Y,X for this factor
    domain_type arguments() const {
      domain_type fd;
      fd.insert(Ydomain_.begin(), Ydomain_.end());
      fd.insert(Xdomain_ptr_->begin(), Xdomain_ptr_->end());
      return fd;
    }

    virtual void print(std::ostream& out) const {
      out << "f[" << Ydomain_ << ", " << (*Xdomain_ptr_) << "]\n";
    }

    // Public methods: Probabilistic queries
    // =========================================================================

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param a  This must assign values to all X in this factor
     *           (but may assign values to any other variables as well).
     * @return  factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    virtual const output_factor_type&
    condition(const input_assignment_type& a) const = 0;

    /**
     * If this factor is f(Y,X), compute f(Y, X = x).
     *
     * @param r Record with values for X in this factor
     *          (which may have values for any other variables as well).
     * @return  factor representing the factor with
     *          the given input variable (X) instantiation;
     *          in real space
     */
    virtual const output_factor_type&
    condition(const input_record_type& r) const = 0;

    /**
     * Returns the empirical expectation of the log of this factor.
     * In particular, if this factor represents P(A|B), then
     * this returns the expected log likelihood of the distribution P(A | B).
     * (But this does not normalize the factor after conditioning.)
     */
    virtual double log_expected_value(const dataset<LA>& ds) const {
      double val(0);
      output_factor_type tmp_fctr;
      double total_ds_weight(0);
      size_t i(0);
      foreach(const record_type& r, ds.records()) {
        tmp_fctr = condition(r);
        val += ds.weight(i) * std::log(tmp_fctr(r));
        total_ds_weight += ds.weight(i);
        ++i;
      }
      assert(total_ds_weight > 0);
      return (val / total_ds_weight);        
    }

    // Public methods: Losses
    // =========================================================================

    //! Compute log P(Y=y|X=x).
    //! @param base  Base of logarithm (default = e).
    double log_likelihood(const record_type& r, double base = exp(1.)) const {
      const output_factor_type& f = condition(r);
      return (f.logv(r) - std::log(f.norm_constant())) / std::log(base);
    }

    //! Returns a functor usable with dataset::expected_value() for computing
    //! expected log likelihood E[log P(X)].
    model_log_likelihood_functor<crf_factor>
    log_likelihood(double base = exp(1.)) const {
      return model_log_likelihood_functor<crf_factor>(*this, base);
    }

    // Public methods: Learning-related methods
    // =========================================================================

    //! @return  true iff the data is stored in log-space
    virtual bool log_space() const = 0;

    //! Tries to change this factor's internal representation to log-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already log-space
    virtual bool convert_to_log_space() = 0;

    //! Tries to change this factor's internal representation to real-space.
    //! This is not guaranteed to work.
    //! @return  true if successful or if the format was already real-space
    virtual bool convert_to_real_space() = 0;

    //! If true, then this is not a learnable factor.
    //! (I.e., the factor's value will be fixed during learning.)
    //! (default after construction = false)
    bool fixed_value() const {
      return fixed_value_;
    }

    //! If true, then this is not a learnable factor.
    //! (I.e., the factor's value will be fixed during learning.)
    //! (default after construction = false)
    //! This returns a mutable reference.
    bool& fixed_value() {
      return fixed_value_;
    }

    //! This tells the factor that the only records it will be conditioned on
    //! will be of this form; this can speed up conditioning for some factors.
    //! This is not guaranteed to be implemented.
    virtual void fix_records(const record_type& r) {
    }

    //! This undoes fix_records().
    virtual void unfix_records() {
    }

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    virtual const optimization_vector& weights() const = 0;

    //! The weights which, along with the feature values, define the factor.
    //! This uses log-space or real-space, whatever is currently set,
    //! but it should only be used with log-space.
    virtual optimization_vector& weights() = 0;

    // Public methods: Other
    // =========================================================================

    /**
     * Relabels outputs Y, inputs X so that
     * inputs may become outputs (if variable_type = output_variable_type) and
     * outputs may become inputs (if variable_type = input_variable_type).
     * The entire argument set must remain the same, i.e.,
     * union(Y,X) must equal union(new_Y, new_X).
     */
    virtual void relabel_outputs_inputs(const output_domain_type& new_Y,
                                        const input_domain_type& new_X) = 0;

    /**
     * Relabels outputs Y, inputs X.
     * The new outputs are new_Y; the new inputs are the remaining variables.
     */
    void relabel_outputs_inputs(const output_domain_type& new_Y) {
      input_domain_type new_X(input_arguments());
      foreach(output_variable_type v, output_arguments()) {
        if (new_Y.count(v) == 0)
          new_X.insert(v);
      }
      relabel_outputs_inputs(new_Y, new_X);
    }

    /**
     * Check the validity of a shuffling of the output, input variables.
     * To be valid, new_Y and new_X must be disjoint,
     * and their union must be a superset of the union of old_Y and old_X.
     * @return True iff valid.
     */
    static bool
    valid_output_input_relabeling(const output_domain_type& old_Y,
                                  const input_domain_type& old_X,
                                  const output_domain_type& new_Y,
                                  const input_domain_type& new_X) {
      domain_type args(old_Y);
      args.insert(old_X.begin(), old_X.end());
      foreach(output_variable_type v, new_Y)
        args.erase(v);
      foreach(input_variable_type v, new_X)
        args.erase(v);
      if (args.size() != 0)
        return false;
      return true;
    }

    // Protected data members
    //==========================================================================
  protected:

    //! Y variables
    output_domain_type Ydomain_;

    //! X variables
    copy_ptr<input_domain_type> Xdomain_ptr_;

    //! If true, then this factor's value stays fixed in crf_parameter_learner.
    //! (default after construction = false)
    bool fixed_value_;

  }; // class crf_factor

  template <typename InputVar, typename OutputFactor, typename OptVector,
            typename LA>
  std::ostream&
  operator<<(std::ostream& out,
             const crf_factor<InputVar, OutputFactor, OptVector, LA>& f) {
    f.print(out);
    return out;
  }

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_CRF_FACTOR_HPP
