#ifndef SILL_MOMENT_GAUSSIAN_HPP
#define SILL_MOMENT_GAUSSIAN_HPP

#include <boost/function.hpp>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/factor/gaussian_base.hpp>
#include <sill/factor/invalid_operation.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/factor/traits.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/logarithmic.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  class canonical_gaussian;

  /**
   * Implementation of a Gaussian factor in the moment form.
   * The factor only implements the sum-product operations.
   *
   * \ingroup factor_types
   */
  class moment_gaussian : public gaussian_base {
  public:
    // DistributionFactor concept types
    typedef boost::function<moment_gaussian(const vector_domain&)> marginal_fn_type;
    typedef boost::function<moment_gaussian(const vector_domain&,
                                            const vector_domain&)> conditional_fn_type;

    // Private data members and methods
    //==========================================================================
  private:
    friend class canonical_gaussian;

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

    //! The multiplicative constant (likelihood)
    logarithmic<double> likelihood;

    /**
     * Initializes the indices for the given arguments and checks
     * matrix dimensions.
     */
    void initialize(const vector_var_vector& head,
                    const vector_var_vector& tail);

    // Constructors and conversion operators
    //==========================================================================
  public:

    //! Constructs a moment Gaussian factor with no arguments
    explicit moment_gaussian(double value = 1)
      : likelihood(value) { }

    //! Constructs a Gaussian for the specified set of variables
    explicit moment_gaussian(const vector_domain& head_list,
                             logarithmic<double> likelihood = 1);

    //! Constructs a Gaussian for the specified sequence of variables
    explicit moment_gaussian(const vector_var_vector& head_list,
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

    moment_gaussian& operator=(const moment_gaussian& other) {
      if (this == &other) {
        return *this;
      }
      if (head_list != other.head_list || tail_list != other.tail_list) {
        head_list = other.head_list;
        tail_list = other.tail_list;
        gaussian_base::operator=(other);
      }
      cmean = other.cmean;
      cov = other.cov;
      coeff = other.coeff;
      likelihood = other.likelihood;
      return *this;
    }

    moment_gaussian& operator=(logarithmic<double> likelihood) {
      *this = moment_gaussian(likelihood); // TODO: optimize this
      return *this;
    }

    // Serialization
    //==========================================================================

    void save(oarchive& ar) const;

    void load(iarchive& ar);

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
      return coeff.n_cols;
    }

    //! Returns the head variables
    const vector_var_vector& head() const {
      return head_list;
    }

    //! Returns the tail variables
    const vector_var_vector& tail() const {
      return tail_list;
    }

    //! Returns the argument sequence of this factor
    vector_var_vector arg_vector() const {
      return concat(head_list, tail_list);
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
      uvec ind = indices(vars);
      return cmean(ind);
    }

    //! Returns the mean for a single variable
    vec mean(vector_variable* v) const {
      return cmean.subvec(safe_get(var_span,v));
    }

    //! Returns the covariance of a subset of variables in the given order
    mat covariance(const vector_var_vector& vars) const {
      uvec ind = indices(vars);
      return cov(ind, ind);
    }

    //! Returns the covariance for a single variable
    mat covariance(vector_variable* v) const {
      return cov(safe_get(var_span, v), safe_get(var_span, v));
    }

    //! Returns the diagonal of the covariance for a subset of variables
    //! in the given order
    vec covariance_diag(const vector_var_vector& vars) const {
      uvec ind(indices(vars));
      vec tmpvec(diagvec(cov));
      return tmpvec(ind);
    }

    //! Returns the covariance of two subsets of variables in the given order:
    //!  cov(vars1, vars2)
    mat covariance(const vector_var_vector& vars1,
                   const vector_var_vector& vars2) const {
      uvec ind1(indices(vars1));
      uvec ind2(indices(vars2));
      return cov(ind1, ind2);
    }

    //! Returns true if the two factors are equivalent
    bool operator==(const moment_gaussian& other) const;

    //! Returns true if the two factors are not equivalent
    bool operator!=(const moment_gaussian& other) const;

    // Factor operations
    //==========================================================================

    //! Evaluates the Gaussian for a given assignment
    logarithmic<double> operator()(const vector_assignment& a) const;

    //! Evaluates the Gaussian P(Y) for a given vector.
    //! (The Gaussian must be a marginal over Y.)
    logarithmic<double> operator()(const vec& y) const;

    //! Evaluates the Gaussian P(Y|X) for given vectors y,x.
    logarithmic<double> operator()(const vec& y, const vec& x) const;

    //! Returns the value associated with an assignment.
    double logv(const vector_assignment& a) const {
      return operator()(a).log_value();
    }

    //! Returns the log-likelihood of this factor given a dataset
    double log_likelihood(const vector_dataset<>& ds) const;

    //! multiplies in another factor
    moment_gaussian& operator*=(const moment_gaussian& x);

    //! multiplies the factor by the given constant
    moment_gaussian& operator*=(logarithmic<double> val);

    //! divides the factor by the given constant
    moment_gaussian& operator/=(logarithmic<double> val);

    //! computes the marginal of the factor over the given variables
    moment_gaussian marginal(const vector_domain& retain) const;

    /**
     * implements Factor::restrict
     * \throws invalid_operation if the covariance matrix over the restricted
     *         variables is singular.
     */
    moment_gaussian restrict(const vector_assignment& a) const;

    //! adds the parameters and the likelihood of another Gaussian
    moment_gaussian& add_parameters(const moment_gaussian& f, double w = 1);
      
    //! implements Factor::subst_args
    moment_gaussian& subst_args(const vector_var_map& map);

    //! implements IndexableFactor::reorder
    moment_gaussian reorder(const vector_var_vector& args) const;

    //! If this factor represents P(A,B|C), then this returns P(A|B,C).
    moment_gaussian conditional(const vector_domain& B) const;

    //! Returns true if this factor represents the conditional p(rest | tail)
    bool is_conditional(const vector_domain& tail) const;

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

    logarithmic<double>& norm_constant() { 
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
      vec vals = zeros(vector_size(head_list));
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
      bool result = arma::chol(At, cov);
      if (!result)
        throw invalid_operation("Cholesky decomposition failed in canonical_gaussian::sample");
//      vals = A * vals + (mg.mean() - (A * vec(A.n_cols, 1.)));
      vals = trans(At) * vals + cmean;
      vector_assignment a;
      size_t k = 0; // index into vals
      foreach(vector_variable* v, head_list) {
        a[v] = vals.subvec(span(k, k + v->size() - 1));
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
    direct_multiplication(const moment_gaussian& x, const moment_gaussian& y);

    friend moment_gaussian
    operator*(const moment_gaussian& x, const moment_gaussian& y);

    friend std::ostream& operator<<(std::ostream&, const moment_gaussian&);

  }; // class moment_gaussian


  // Free functions
  //============================================================================

  //! \relates moment_gaussian
  std::ostream& operator<<(std::ostream& out, const moment_gaussian& mg);
  
  //! \relates moment_gaussian
  moment_gaussian operator*(const moment_gaussian& x, const moment_gaussian& y);

  //! \relates moment_gaussian
  inline moment_gaussian operator/(moment_gaussian x, logarithmic<double> a) {
    return x /= a;
  }

  //! \relates moment_gaussian
  double norm_inf(const moment_gaussian& x, const moment_gaussian& y);

  // Utility classes
  //============================================================================
  typedef boost::function<moment_gaussian(const vector_domain&)>
    marginal_moment_gaussian_fn;

  typedef boost::function<moment_gaussian(const vector_domain&,
                                          const vector_domain&)>
    conditional_moment_gaussian_fn;

  // Traits
  //============================================================================

  //! \addtogroup factor_traits
  //! @{

  template <>
  struct has_multiplies<moment_gaussian> : public boost::true_type { };

  template <>
  struct has_multiplies_assign<moment_gaussian> : public boost::true_type { };

  template <>
  struct has_marginal<moment_gaussian> : public boost::true_type { };

  //! @}
  
} // namespace sill

#include <sill/macros_undef.hpp>

#include <sill/factor/gaussian_common.hpp>
#include <sill/factor/util/operations.hpp>

#endif 
