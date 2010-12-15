#ifndef PRL_MOMENT_GAUSSIAN_HPP
#define PRL_MOMENT_GAUSSIAN_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/lagged_fibonacci.hpp>

#include <prl/factor/constant_factor.hpp>
#include <prl/factor/gaussian_factor.hpp>
#include <prl/factor/invalid_operation.hpp>
#include <prl/math/linear_algebra.hpp>
#include <prl/math/logarithmic.hpp>
#include <prl/math/matrix.hpp>
#include <prl/math/vector.hpp>
#include <prl/range/forward_range.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  class canonical_gaussian;
  //  class gaussian_factor;

  /**
   * Implementation of a Gaussian factor in the moment form.
   * The factor only implements the sum-product operations.
   *
   * \ingroup factor_types
   */
  class moment_gaussian : public gaussian_factor {
    friend class canonical_gaussian;

    // Public type declarations
    //==========================================================================
  public:
    //! implements Factor::collapse_type
    typedef moment_gaussian collapse_type;

    //! implements Factor::collapse_ops
    static const unsigned collapse_ops = 1 << sum_op;

    //! implements Factor::combine_ops
    static const unsigned combine_ops = 1 << product_op | 1 << divides_op;

    // Private data members
    //==========================================================================
  private:
    //! A list of head arguments in their natural order
    vector_var_vector head_list;

    //! A list of tail arguments in their natural order
    vector_var_vector tail_list;

    //! The (conditional) mean
    vec cmean;

    //! The (conditional) covariance
    mat cov;

    //! The matrix of coefficients (size = size_head x size_tail)
    mat coeff;

  public:
    //! The multiplicative constant (likelihood)
    logarithmic<double> likelihood;


    //! Serialize / deserialize members
    void serialize(oarchive & ar) const;
    void deserialize(iarchive & ar);

  private:

    /**
     * Initializes the indices for the given arguments and checks
     * matrix dimensions.
     */
    void initialize(const vector_var_vector& head,
                    const vector_var_vector& tail);

    friend moment_gaussian combine(const moment_gaussian& x,
                                   const moment_gaussian& y,
                                   op_type op);

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Constructs a moment Gaussian factor with no arguments
    moment_gaussian(double value = 1)
      : likelihood(value) { }

    //! Constructs a Gaussian for the specified collection of variables
    explicit moment_gaussian(const vector_domain& head_list,
                             logarithmic<double> likelihood = 1);

    //! Constructs a Gaussian which will represent a conditional distribution
    //! for the specified collection of variables
    moment_gaussian(const vector_var_vector& head_list,
                    const vector_var_vector& tail_list,
                    logarithmic<double> likelihood = 1);

    /**
     * Constructs a Gaussian factor that represents a marginal distribution.
     * \param args  a sequence of variables
     * \param cmean the mean; defaults to zeros
     * \param cov   the covariance; defaults to identity
     */
    moment_gaussian(const vector_var_vector& head_list,
                    const vec& cmean,
                    const mat& cov = mat(),
                    logarithmic<double> likelihood = 1);

    /**
     * Constructs a Gaussian factor that represents a conditional distribution.
     */
    moment_gaussian(const vector_var_vector& head_list,
                    const vec& cmean,
                    const mat& cov,
                    const vector_var_vector& tail_list,
                    const mat& coeff,
                    logarithmic<double> likelihood = 1);

    /**
     * Converts a canonical Gaussian into the moment form representation.
     * The canonical Gaussian must represent a marginal (not conditional)
     * distribution.
     */
    explicit moment_gaussian(const canonical_gaussian& cg);

    //! Conversion from constant_factor
    moment_gaussian(const constant_factor& factor);

    /**
     * Conversion to a constant factor. The conversion is only supported
     * when the factor has no arguments; otherwise, a run-time assertion
     * is thrown.
     * @see a note on conversion operators in table_factor::o()
     */
    operator constant_factor() const;

    //! conversion to human-readable representation
    operator std::string() const;

    // Accessors
    //==========================================================================

    //! Returns the total number of dimensions
    size_t size() const {
      return size_head() + size_tail();
    }

    //! Returns the total vector size (length) of the head variables
    size_t size_head() const {
      return cmean.size();
    }

    //! Returns the total vector size (length) of the tail variables
    size_t size_tail() const {
      return coeff.size2();
    }

    //! Returns the head variables
    const vector_var_vector& head() const {
      return head_list;
    }

    //! Returns the tail variables
    const vector_var_vector& tail() const {
      return tail_list;
    }

    //! Returns true if the factor represents a marginal distribution
    bool marginal() const {
      return tail_list.empty();
    }

    //! Returns the mean vector of the factor (in the natural order)
    const vec& mean() const {
      return cmean;
    }

    //! Returns the mean vector of the factor
    //! The caller must not alter the dimensions of the vector.
    vec& mean() {
      return cmean;
    }

    //! Returns the covariance matrix of the factor in the natural order.
    const mat& covariance() const {
      return cov;
    }

    //! Returns the covariance matrix of the factor in the natural order.
    //! The caller must not alter the dimensions of the matrix.
    mat& covariance() {
      return cov;
    }
    
    //! Returns the coefficients of a conditional distribution
    const mat& coefficients() const {
      return coeff;
    }

    //! Returns the coefficients of a conditional distribution
    //! The caller must not alter the dimensions of the matrix.
    mat& coefficients() {
      return coeff;
    }

    //! Returns the mean over a subset of variables in the given order
    vec mean(const vector_var_vector& vars) const {
      ivec ind = indices(vars);
      return cmean(ind);
    }

    //! Returns the mean for a single variable
    vec mean(vector_variable* v) const {
      return cmean(safe_get(var_range,v));
    }

    //! Returns the covariance a subset of variables in the given order
    mat covariance(const vector_var_vector& vars) const {
      ivec ind = indices(vars);
      return cov(ind, ind);
    }

    //! Returns the covariance for a single variable
    mat covariance(vector_variable* v) const {
      return cov(safe_get(var_range, v), safe_get(var_range, v));
    }

    //! Returns the diagonal of the covariance for a subset of variables
    //! in the given order
    vec covariance_diag(const vector_var_vector& vars) const {
      ivec ind(indices(vars));
      vec tmpvec(diag(cov));
      return tmpvec(ind);
    }

    //! Returns true if the two factors are equivalent
    bool operator==(const moment_gaussian& other) const;

    //! Returns true if the two factors are not equivalent
    bool operator!=(const moment_gaussian& other) const;

    // Factor operations
    //==========================================================================
    //! Evaluates the Gaussian for a given assignment
    logarithmic<double> operator()(const vector_assignment& a) const;

    //! Evaluates the Gaussian for a given vector
    logarithmic<double> operator()(const vec& x) const;

    //! Returns the value associated with an assignment.
    double logv(const vector_assignment& a) const {
      return operator()(a).log_value();
    }

    //! implements Factor::combine_in for multiplication
    moment_gaussian& combine_in(const moment_gaussian& x, op_type op);

    //! multiplies or divides the factor by the given constant
    moment_gaussian& combine_in(const constant_factor& x, op_type op);

    //! implements Factor::collapse
    moment_gaussian collapse(const vector_domain& retain, op_type op) const;

    /**
     * implements Factor::restrict
     * \throws invalid_operation if the covariance matrix over the restricted
     *         variables is singular.
     * @todo Currently, if the factor has any tail variables,
     *       then all of them must be restricted. Permit more flexibility.
     */
    moment_gaussian restrict(const vector_assignment& a) const;

    //! adds the parameters and the likelihood of another Gaussian
    moment_gaussian& add_parameters(const moment_gaussian& f, double w = 1);
      
    //! implements Factor::subst_args
    moment_gaussian& subst_args(const vector_var_map& map);

    //! implements DistributionFactor::marginal
    moment_gaussian marginal(const vector_domain& retain) const {
      return collapse(retain, sum_op);
    }

    //! If this factor represents P(A,B), then this returns P(A|B).
    //! This may only be called on marginal distributions.
    moment_gaussian conditional(const vector_domain& B) const;

    //! implements DistributionFactor::is_normalizable
    bool is_normalizable() const {
      return true;  // FIXME is_positive_finite(norm_constant()); 
    }

    //! implements DistributionFactor::normalize
    moment_gaussian& normalize() {
      // assert(is_finite(log_lik)); FIXME
      likelihood = 1;
      return *this;
    }

    //! implements DistributionFactor::norm_constant
    logarithmic<double> norm_constant() const { 
      return likelihood;
    }

    //! Returns a sample from the factor, which is assumed to be normalized
    //! to be a distribution P(arguments).
    //! WARNING: Not all random number generators work with this; for example,
    //!  boost::mt11213b does not work, but boost::lagged_fibonacci607 works.
    //!  If this is given a wrong generator, it will use the given generator
    //!  to choose a random seed for a workable generator.
    template <typename RandomNumberGenerator>
    vector_assignment sample(RandomNumberGenerator& rng) const {
      if (!marginal())
        throw std::runtime_error
          ("moment_gaussian::sample() was called on a conditional Gaussian.");
      // Sample vals ~ Normal(1's, Identity),
      // i.e., the multivariate standard normal distribution.
      vec vals(vector_size(head_list), 0.);
      if (vals.size() == 0)
        return vector_assignment();
      boost::normal_distribution<double> normal_dist(0,1);
      bool use_different_rng(false);
      foreach(double& val, vals) {
        val = normal_dist(rng);
        if (std::isnan(val)) {
          use_different_rng = true;
          break;
        }
      }
      if (use_different_rng) {
        boost::lagged_fibonacci607
          newrng(boost::uniform_int<int>
                 (0, std::numeric_limits<int>::max())(rng));
        vals[0] = normal_dist(newrng); //avoid bug in boost::normal_distribution
        foreach(double& val, vals)
          val = normal_dist(newrng);
      }
      // Transform vals to be sampled from this Gaussian distribution.
      mat At;
      bool result = chol(cov, At);
      if (!result)
        throw invalid_operation("Cholesky decomposition failed in canonical_gaussian::sample");
//      vals = A * vals + (mg.mean() - (A * vec(A.size2(), 1.)));
      vals = At.transpose() * vals + cmean;
      vector_assignment a;
      size_t k(0); // index into vals
      foreach(vector_variable* v, head_list) {
        a[v] = vals(irange(k, k + v->size()));
        k += v->size();
      }
      return a;
    }

    //! implements DistributionFactor::entropy
    //! This should only be called for a marginal distribution.
    double entropy(double base) const;

    //! implements DistributionFactor::entropy
    //! This should only be called for a marginal distribution.
    //! Uses base e logarithm.
    double entropy() const;

    //! implements DistributionFactor::relative_entropy
    double relative_entropy(const moment_gaussian& q) const;

    //! Computes the mutual information between two sets of variables
    //! in this factor's arguments. The sets of f1, f2 must be disjoint.
    //! Note: This factor must be a marginal distribution.
    double mutual_information(const vector_domain& d1,
                              const vector_domain& d2) const;

  private:
    /**
     * Multiplies together two moment_gaussian factors.  The head of y
     * must be disjoint from the domain of x, and x must be a marginal
     * distribution.  For a more complete algorithm, see Sec. 4.4 of
     * Lauritzen & Jensen (1999).
     */
    static moment_gaussian 
    direct_combination(const moment_gaussian& x, const moment_gaussian& y);

  }; // class moment_gaussian

  //! \relates moment_gaussian
  std::ostream& operator<<(std::ostream& out, const moment_gaussian& mg);


  // Free functions
  //============================================================================
  
  //! \relates moment_gaussian
  moment_gaussian combine(const moment_gaussian& x,
                          const moment_gaussian& y,
                          op_type op);

  template<> struct combine_result<moment_gaussian, moment_gaussian> {
    typedef moment_gaussian type;
  };

} // namespace prl

#include <prl/macros_undef.hpp>

#include <prl/factor/operations.hpp>

#endif 
