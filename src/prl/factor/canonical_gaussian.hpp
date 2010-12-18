#ifndef SILL_CANONICAL_GAUSSIAN_HPP
#define SILL_CANONICAL_GAUSSIAN_HPP

#include <sill/base/universe.hpp>
#include <sill/factor/constant_factor.hpp>
#include <sill/factor/gaussian_factor.hpp>
#include <sill/factor/invalid_operation.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/dataset/vector_record.hpp>
#include <sill/math/linear_algebra.hpp>

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

    // Public type declarations
    //==========================================================================
  public:
    //! implements Factor::collapse_type
    typedef canonical_gaussian collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = 1 << sum_op;

    //! implements Factor::combine_ops
    static const unsigned combine_ops = 1 << product_op | 1 << divides_op;

    // Private data members
    //==========================================================================
  private:
    //! A list of arguments in their natural order
    vector_var_vector arg_list;

    //! The information matrix
    mat lambda;

    //! The information vector
    vec eta;

  public:
    //! The multiplicative constant
    double log_mult;

    void save(oarchive& ar) const;

    void load(iarchive& ar);

  private:

    /**
     * Initializes this indices for the given arguments and checks
     * matrix dimensions.
     * \paran use_default
     *        if true, resets the information matrix/vector to zero.
     */
    void initialize(const forward_range<vector_variable*>& args,
                    bool use_default);

    friend canonical_gaussian combine(const canonical_gaussian& x,
                                      const canonical_gaussian& y,
                                      op_type op);

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Constructs a canonical Gaussian factor with no arguments
    canonical_gaussian(double value = 1) : log_mult(std::log(value)) { }

    /**
     * Constructs a canonical Gaussian factor with a given set of
     * arguments and zero parameters.
     */
    canonical_gaussian(const vector_domain& args, double value);

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

    //! Conversion from constant_factor
    canonical_gaussian(const constant_factor& factor);

    //! Conversion from a moment_gaussian
    canonical_gaussian(const moment_gaussian& mg);

    //! Assignment operator.
    //    canonical_gaussian& operator=(const canonical_gaussian& other);

    /**
     * Converts to a constant factor. The conversion is only supported
     * when the factor has no arguments; otherwise, a run-time assertion
     * is thrown.
     * @see a note on conversion operators in table_factor::o()
     */
    operator constant_factor() const;

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

    // Accessors
    //==========================================================================

    //! Returns the argument list of this Gaussian
    const vector_var_vector& argument_list() const {
      return arg_list;
    }

    //! Returns the number of dimensions of this Gaussian
    size_t size() const {
      return eta.size();
    }

    //! Returns the information matrix in the natural order
    const mat& inf_matrix() const {
      return lambda;
    }
    
    //! Returns the information matrix in the natural order
    //! The caller must not alter the matrix dimensions
    mat& inf_matrix() {
      return lambda;
    }

    //! Returns the information vector in the natural order
    const vec& inf_vector() const { 
      return eta;
    }

    //! Returns the information vector in the natural order
    //! The caller must not alter the vector dimensions
    vec& inf_vector() {
      return eta;
    }

    //! Returns the log multiplier.
    double log_multiplier() const {
      return log_mult;
    }

    //! Returns the log multiplier.
    double& log_multiplier() {
      return log_mult;
    }

    //! Returns the information matrix for a subset of the arguments
    mat inf_matrix(const vector_var_vector& args) const {
      ivec ind(indices(args));
      return lambda(ind, ind);
    }

    //! Returns the information vector for a subset of the arguments
    vec inf_vector(const vector_var_vector& args) const {
      ivec ind(this->indices(args));
      return eta(ind);
    }

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
    logarithmic<double> operator()(const vector_record& r) const;

    //! Returns the log-likelihood of the factor
    double logv(const vector_assignment& a) const;

    //! Returns the log-likelihood of the factor
    double logv(const vector_record& r) const;

    /**
     * implements Factor::collapse
     * \throws invalid_operation if the information matrix over the retained
     *         variables is singular.
     * \todo fix the log-likelihood
     */
    canonical_gaussian collapse(const vector_domain& retain, op_type op) const;
    canonical_gaussian collapse(op_type op, const vector_domain& retain) const;

    //! Performs a collapse operation, storing result in factor cg.
    //! Avoids reallocation if possible.
    void collapse(canonical_gaussian& cg, const vector_domain& retain,
                  op_type op) const;

    //! implements Factor::restrict
    canonical_gaussian restrict(const vector_assignment& a) const;
  
    //! implements Factor::subst_args
    canonical_gaussian& subst_args(const vector_var_map& map);

    //! implements DistributionFactor::marginal
    canonical_gaussian marginal(const vector_domain& retain) const {
      return collapse(retain, sum_op);
    }

    //! Computes marginal, storing result in factor f.
    //! If f is pre-allocated, this avoids reallocation.
    void marginal(canonical_gaussian& cg, const vector_domain& retain) const {
      collapse(cg, retain, sum_op);
    }

    //! If this factor represents P(A,B), then this returns P(A|B).
    //! @todo Make this more efficient.
    canonical_gaussian conditional(const vector_domain& B) const;

    //! implements DistributionFactor::is_normalizable
    bool is_normalizable() const {
      return true;
    }

    //! Returns the normalization constant
    double norm_constant() const;

    //! Returns the log of the normalization constant
    double log_norm_constant() const;

    //! implements Distribution::normalize
    canonical_gaussian& normalize();

    //! Computes the maximum for each assignment to the given variables
    canonical_gaussian maximum(const vector_domain& retain) const;

    //! Returns an assignment that achieves the maximum value (i.e., the mean).
    //! @todo Move free functions into this class (and same for other factors).
    vector_assignment arg_max() const;

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

    // Factor operations: combining factors
    //==========================================================================

    //! implements Factor::combine_in
    canonical_gaussian& combine_in(const canonical_gaussian& x, op_type op);

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

  }; // class canonical_gaussian

  //! \relates canonical_gaussian
  std::ostream& operator<<(std::ostream& out, const canonical_gaussian& cg);


  //! Combines two canonical Gaussian factors
  //! \relates canonical_gaussian
  canonical_gaussian combine(const canonical_gaussian& x,
                             const canonical_gaussian& y,
                             op_type op);

  // TODO: this line should be below (but can't)
  // either remove the default implementation of combine(F,F)
  // or add a default template specialization fo combine_result<F,F>
  template<> struct combine_result<canonical_gaussian, canonical_gaussian> {
    typedef canonical_gaussian type;
  };

  //! Combines a moment and a canonical Gaussian factor
  //! \relates canonical_gaussian
  inline canonical_gaussian combine(const moment_gaussian& mg,
                                    const canonical_gaussian& cg,
                                    op_type op) {
    return combine(canonical_gaussian(mg), cg, op);
  }
  
  //! Combines a moment and a canonical Gaussian factor
  //! \relates canonical_gaussian
  inline canonical_gaussian combine(const canonical_gaussian& cg,
                                    const moment_gaussian& mg,
                                    op_type op) {
    return combine(cg, canonical_gaussian(mg), op);
  }
  
  template<> struct combine_result<moment_gaussian, canonical_gaussian> {
    typedef canonical_gaussian type;
  };

  template<> struct combine_result<canonical_gaussian, moment_gaussian> {
    typedef canonical_gaussian type;
  };
  
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

} // namespace sill

#include <sill/macros_undef.hpp>

#include <sill/factor/operations.hpp>

#endif
