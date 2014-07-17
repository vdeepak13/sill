#ifndef SILL_CANONICAL_GAUSSIAN_HPP
#define SILL_CANONICAL_GAUSSIAN_HPP

#include <sill/base/universe.hpp>
#include <sill/factor/gaussian_factor.hpp>
#include <sill/factor/invalid_operation.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/operations.hpp>
#include <sill/factor/traits.hpp>
#include <sill/learning/dataset/vector_record.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  class moment_gaussian;

  /**
   * A Gaussian factor in the natural (canonical) form. 
   * The factor supports the sum-product operations.
   *
   * \ingroup factor_types
   */
  class canonical_gaussian : public gaussian_factor {

    friend class moment_gaussian;

    // Constructors and conversion operators
    //==========================================================================
  public:

    //! Constructs a canonical Gaussian factor with no arguments
    explicit canonical_gaussian(double value = 1) : log_mult(std::log(value)) { }

    /**
     * Constructs a canonical Gaussian factor with a given set of
     * arguments and zeroed parameters.
     */
    explicit canonical_gaussian(const vector_domain& args);

    /**
     * Constructs a canonical Gaussian factor with a given set of
     * arguments and zero parameters.
     */
    canonical_gaussian(const vector_domain& args, double value);

    /**
     * Constructs a canonical Gaussian factor with a given set of
     * arguments and zeroed parameters, using the given variable ordering.
     */
    explicit canonical_gaussian(const vector_var_vector& args);

    /**
     * Constructs a canonical Gaussian factor with a given set of
     * arguments and zero parameters, using the given variable ordering.
     */
    canonical_gaussian(const vector_var_vector& args, double value);

    /**
     * Constructs a canonical Gaussian factor with a given set of
     * arguments and zero parameters, using the given variable ordering.
     */
    canonical_gaussian(const forward_range<vector_variable*>& args,
                       double value);

    /**
     * Constructs a canonical Gaussian factor with a given sequence of
     * arguments.
     * \param args a sequence of variables
     * \param lambda the information matrix
     * \param eta the information vector
     */
    canonical_gaussian(const vector_var_vector& args,
                       const mat& lambda,
                       const vec& eta,
                       double log_mult = 0);

    //! Conversion from a moment_gaussian
    canonical_gaussian(const moment_gaussian& mg);

    //! conversion to human-readable representation
    operator std::string() const;

    /**
     * Mimics a constructor, but resets this factor rather than creating a
     * new factor.
     * \param args a sequence of variables
     * \param lambda the information matrix
     * \param eta the information vector
     */
    void reset(const vector_var_vector& args, const mat& lambda, const vec& eta,
               double log_mult = 0);

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const;

    void load(iarchive& ar);

    // Accessors
    //==========================================================================

    //! Returns the argument list of this Gaussian
    const vector_var_vector& argument_list() const;

    //! Returns the number of dimensions of this Gaussian
    size_t size() const;

    //! Returns the information matrix in the natural order
    const mat& inf_matrix() const;
    
    //! Returns the information matrix in the natural order
    //! The caller must not alter the matrix dimensions
    mat& inf_matrix();

    //! Returns the information vector in the natural order
    const vec& inf_vector() const;

    //! Returns the information vector in the natural order
    //! The caller must not alter the vector dimensions
    vec& inf_vector();

    //! Returns the log multiplier.
    double log_multiplier() const;

    //! Returns the log multiplier.
    double& log_multiplier();

    //! Returns the information matrix for a subset of the arguments
    mat inf_matrix(const vector_var_vector& args) const;

    //! Returns the information vector for a subset of the arguments
    vec inf_vector(const vector_var_vector& args) const;

    // Comparison operators
    //==========================================================================

    //! Returns true of the two factors have the same argument set
    //! and are equivalent.
    bool operator==(const canonical_gaussian& other) const;

    //! Returns true if the two factors are not equivalent
    bool operator!=(const canonical_gaussian& other) const;

    //! Returns true if the first factor precedes the second in the
    //! lexicographical ordering.
    bool operator<(const canonical_gaussian& other) const;

    // Factor operations
    //==========================================================================

    //! Evaluates the factor for an assignment
    //! \todo this function needs to be tested
    logarithmic<double> operator()(const vector_assignment& a) const;

    //! Evaluates the factor for a record.
    logarithmic<double> operator()(const record_type& r) const;

    //! Returns the log-likelihood of the factor
    double logv(const vector_assignment& a) const;

    //! Returns the log-likelihood of the factor
    double logv(const record_type& r) const;

    //! multiplies in another factor
    canonical_gaussian& operator*=(const canonical_gaussian& x);

    //! divides by another factor
    canonical_gaussian& operator/=(const canonical_gaussian& x);

    //! multiplies the factor by the given constant
    canonical_gaussian& operator*=(logarithmic<double> val);

    //! divides the factor by the given constant
    canonical_gaussian& operator/=(logarithmic<double> val);

    /**
     * computes marginal over a subset of variables
     * \throws invalid_operation if the information matrix over the retained
     *         variables is singular.
     */
    canonical_gaussian marginal(const vector_domain& retain) const;

    //! Computes marginal, storing result in the given factor.
    //! Avoids reallocation if possible.
    void marginal(const vector_domain& retain, canonical_gaussian& cg) const;

    //! Computes marginal, storing result in the given factor.
    //! Avoids reallocation if possible.
    //! This version does not update the normalization constant.
    void marginal_unnormalized(const vector_domain& retain,
                               canonical_gaussian& cg) const;

    //! Computes the maximum for each assignment to the given variables
    canonical_gaussian maximum(const vector_domain& retain) const;

    //! Returns an assignment that achieves the maximum value (i.e., the mean).
    //! @todo Move free functions into this class (and same for other factors).
    vector_assignment arg_max() const;

    //! Restricts (conditions) the variable for the given assignment
    canonical_gaussian restrict(const vector_assignment& a) const;
  
    /**
     * Restrict which stores the result in the given factor f.
     * TO DO: Avoid reallocation if f has been pre-allocated.
     *
     * @param r_vars  Only restrict away arguments of this factor which
     *                appear in both keys(r) and r_vars.
     */
    void restrict(const record_type& r, const vector_domain& r_vars,
                  canonical_gaussian& f) const;

    /**
     * Restrict which stores the result in the given factor f.
     * TO DO: Avoid reallocation if f has been pre-allocated.
     *
     * @param r_vars  Only restrict away arguments of this factor which
     *                appear in both keys(r) and r_vars.
     * @param strict  Require that all variables which are in
     *                intersect(f.arguments(), r_vars) appear in keys(r).
     */
    void restrict(const record_type& r, const vector_domain& r_vars,
                  bool strict, canonical_gaussian& f) const;
  
    //! implements Factor::subst_args
    canonical_gaussian& subst_args(const vector_var_map& map);

    //! If this factor represents P(A,B), then this returns P(A|B).
    //! @todo Make this more efficient.
    canonical_gaussian conditional(const vector_domain& B) const;

    //! Returns true if this factor represents the conditional p(rest | tail)
    //! @return true for now (TODO: figure out if we can test this)
    bool is_conditional(const vector_domain& tail) const { return true; }
    
    //! implements DistributionFactor::is_normalizable
    bool is_normalizable() const;

    //! Returns the normalization constant
    double norm_constant() const;

    //! Returns the log of the normalization constant
    double log_norm_constant() const;

    //! implements Distribution::normalize
    canonical_gaussian& normalize();

    /**
     * Returns a sample from the factor, which is assumed to be normalized
     * to be a distribution P(arguments).
     *
     * NOTE: If you take multiple samples, it is better to convert to a
     *       moment_gaussian and then sample.
     *
     * WARNING: Not all random number generators work with this; for example,
     *  boost::mt11213b does not work, but boost::lagged_fibonacci607 works.
     *  If this is given a wrong generator, it will use the given generator
     *  to choose a random seed for a workable generator.
     *
     * @tparam RandomNumGen  Random number generator.
     */
    template <typename RandomNumGen>
    vector_assignment sample(RandomNumGen& rng) const {
      moment_gaussian mg(*this);
      return mg.sample(rng);
    }

    //! implements Distribution::entropy
    double entropy(double base) const;

    //! implements Distribution::entropy
    //! Uses base e logarithm.
    double entropy() const;

    //! implements Distribution::relative_entropy
    double relative_entropy(const canonical_gaussian& q) const;

    //! Computes the mutual information between two sets of variables
    //! in this factor's arguments. The sets of f1, f2 must be disjoint.
    //! Note: This factor must be a marginal distribution.
    double mutual_information(const vector_domain& d1,
                              const vector_domain& d2) const;

    // Other operations
    //==========================================================================

    /**
     * Ensures that the information matrix is PSD, i.e., this factor represents
     * a valid likelihood.
     * \param mean the mean of the joint distribution
     * @return  True if the information matrix was already PSD and false if
     *          it was not and was adjusted.
     */
    bool enforce_psd(const vec& mean);

    // Private data members
    //==========================================================================
  private:
    //! A list of arguments in their natural order
    vector_var_vector arg_list;

    //! The information matrix
    mat lambda;

    //! The information vector
    vec eta;

    //! The multiplicative constant
    double log_mult;

    // Private methods
    //==========================================================================

    /**
     * Initializes this indices for the given arguments and checks
     * matrix dimensions.
     * \paran use_default
     *        if true, resets the information matrix/vector to zero.
     */
    void initialize(const forward_range<vector_variable*>& args,
                    bool use_default);

    void marginal(const vector_domain& retain,
                  bool renormalize,
                  canonical_gaussian& cg) const;

    canonical_gaussian& combine_in(const canonical_gaussian& x, double sign);
    
    friend canonical_gaussian combine(const canonical_gaussian& x,
                                      const canonical_gaussian& y,
                                      double sign);


  }; // class canonical_gaussian
  
  //! \relates canonical_gaussian
  std::ostream& operator<<(std::ostream& out, const canonical_gaussian& cg);

  // Mulitplication and division
  //==========================================================================

  //! \relates canonical_gaussian
  canonical_gaussian 
  operator*(const canonical_gaussian& x, const canonical_gaussian& y);

  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator*(const moment_gaussian& mg, const canonical_gaussian& cg) {
    return canonical_gaussian(mg) * cg;
  }

  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator*(const canonical_gaussian& cg, const moment_gaussian& mg) {
    return cg * canonical_gaussian(mg);
  }

  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator*(canonical_gaussian cg, logarithmic<double> x) {
    return cg *= x;
  }

  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator*(logarithmic<double> x, canonical_gaussian cg) {
    return cg *= x;
  }
  
  //! \relates canonical_gaussian
  canonical_gaussian
  operator/(const canonical_gaussian& x, const canonical_gaussian& y);

  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator/(const moment_gaussian& mg, const canonical_gaussian& cg) {
    return canonical_gaussian(mg) / cg;
  }

  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator/(const canonical_gaussian& cg, const moment_gaussian& mg) {
    return cg / canonical_gaussian(mg);
  }

  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator/(canonical_gaussian cg, logarithmic<double> x) {
    return cg /= x;
  }
  
  //! \relates canonical_gaussian
  inline canonical_gaussian
  operator/(logarithmic<double> x, const canonical_gaussian& cg) {
    return canonical_gaussian(x) /= cg;
  }
  
  // Other functions
  //==========================================================================

  //! Computes the L-infinity norm of the parameters of two canonical Gaussians
  //! \relates canonical_gaussian
  double norm_inf(const canonical_gaussian& x, const canonical_gaussian& y);

  //! Exponentiates a canonical Gaussian 
  //! (which is equivalent to multiplying the parameters by a constant)
  canonical_gaussian pow(const canonical_gaussian& cg, double a);

  //! Returns an assignment that achieves the maximum value (i.e., the mean).
  vector_assignment arg_max(const canonical_gaussian& cg);

  //! Returns \f$f_1^{(1-a)} * f_2^a\f$
  canonical_gaussian weighted_update(const canonical_gaussian& f1,
                                     const canonical_gaussian& f2,
                                     double a);
  
  //! Returns the inverse of the factor (flips the sign on information vec & mat)
  canonical_gaussian invert(const canonical_gaussian& f);

  // Traits
  //============================================================================

  //! \addtogroup factor_traits
  //! @{

  template <>
  struct has_multiplies<canonical_gaussian> : public boost::true_type { };

  template <>
  struct has_multiplies_assign<canonical_gaussian> : public boost::true_type { };

  template <>
  struct has_divides<canonical_gaussian> : public boost::true_type { };

  template <>
  struct has_divides_assign<canonical_gaussian> : public boost::true_type { };

  template <>
  struct has_marginal<canonical_gaussian> : public boost::true_type { };

  template <>
  struct has_maximum<canonical_gaussian> : public boost::true_type { };

  template <>
  struct has_arg_max<canonical_gaussian> : public boost::true_type { };

  //! @}
  
} // namespace sill

#include <sill/macros_undef.hpp>

#include <sill/factor/operations.hpp>

#endif
