#ifndef SILL_NONLINEAR_GAUSSIAN_HPP
#define SILL_NONLINEAR_GAUSSIAN_HPP

#include <boost/shared_ptr.hpp>

#include <sill/factor/factor.hpp>
#include <sill/factor/traits.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/approx/interfaces.hpp>
#include <sill/math/function/interfaces.hpp>

namespace sill {

  class gaussian_approximator;
  class moment_gaussian;

  /**
   * A factor that represents a conditional nonlinear Gaussian distribution.
   * The distribution is of the form Y | x ~ N(f(x), Sigma).
   * When a factor if this type is multiplication with a Gaussian factor
   * that represents a marginal over (a superset of) the tail variables,
   * the joint product is approximated as a Gaussian.
   *
   * This factor type cannot be directly collapsed. At the moment, the factor
   * cannot be combined with other factors of the same type (that is, the 
   * nonlinear Gaussian factors cannot be chained).
   * 
   * \ingroup factor_types
   * \see Factor
   */
  class nonlinear_gaussian : public factor {
    
    // Public type declarations 
    //==========================================================================
  public: 
    //! implements Factor::result_type
    typedef double result_type; // TODO (Stano): is this correct?
    
    //! implements Factor::variable_type
    typedef vector_variable variable_type;

    //! implements Factor::domain_type
    typedef vector_domain domain_type;

    //! implements Factor::assignment_Type
    typedef vector_assignment assignment_type;

    // Private data members and accessors
    //==========================================================================
  private:
    //! The arguments
    vector_domain args;

    //! The head arguments; (always unrestricted)
    vector_var_vector head;

    //! The tail arguments (only those that have not been restricted)
    vector_var_vector tail;

    //! The function that computes the conditional mean of head, given the tail
    boost::shared_ptr< vector_function > fmean_ptr;
    
    //! A mapping from the factor input to the function input
    std::vector<size_t> input_map;

    //! The values for the restricted variables. 
    //! fixed_input[i] holds the value for each i that was fixed.
    vec fixed_input;

    //! The restricted arguments of the head
    vector_assignment assignment_;

    //! The conditional covariance
    mat cov;

    //! An object that approximates the joint
    boost::shared_ptr< gaussian_approximator > approx_ptr;

    //! Returns the function associated with this CPD
    const vector_function& fmean() const {
      return *fmean_ptr;
    }

    //! Returns the object that approximates the joint
    const gaussian_approximator& approx() const {
      return *approx_ptr;
    }

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Creates a CPD with no head and no tail
    nonlinear_gaussian() { }
    
    //! Creates a conditional nonlinear Guassian CPD with the specified
    //! head and tail variables, mean function, and covariance
    nonlinear_gaussian(const vector_var_vector& head,
                       const vector_var_vector& tail,
                       const vector_function& fmean,
                       const gaussian_approximator& approx,
                       const mat& cov = mat());

    //! Conversion to human-readable representation
    operator std::string() const;

    // Accessors
    //==========================================================================
    //! Returns the argument set of this factor
    const vector_domain& arguments() const {
      return args;
    }

    //! Returns the list of head arguments
    const vector_var_vector& head_list() const {
      return head;
    }
    
    //! Returns the list of tail arguments
    const vector_var_vector& tail_list() const {
      return tail;
    }

    //! Returns the dimensionality of the head arguments
    size_t size_head() const {
      return fmean().size_out();
    }

    //! Returns the dimensionality of the tail arguments
    size_t size_tail() const {
      return vector_size(tail);
    }

    //! Returns the total dimensionality, head and tail including, 
    //! of the arguments
    size_t size() const {
      return size_head() + size_tail();
    }

    //! Returns the conditional covariance
    const mat& covariance() const {
      return cov;
    }

    //! Returns the assignment that restricts the head variables
    const vector_assignment& assignment() const {
      return assignment_;
    }

    // Queries
    //==========================================================================
    //! Returns the Gaussian approximation of this factor with respect to 
    //! a Gaussian prior.
    canonical_gaussian approximate(const moment_gaussian& prior) const;

    //! Evaluates the conditional mean for the specified input
    vec mean(const vec& input) const;

    // Factor operations
    //==========================================================================
    //! Evaluates the conditional likelihood for the specified input
    vec operator()(const vec& input) const {
      assert(false); // TODO
    }

    //! Evaluates the log-likelihood for the specified input
    vec log_likelihood(const vec& input) const {
      assert(false); // TODO
    }
    
    //! implements Factor::operator()
    double operator()(const vector_assignment& a) const {
      assert(false); // TODO
      return 0;
    }
    
    //! restricts (conditions) the factor to an assignment
    //! \todo this function needs to be tested
    nonlinear_gaussian restrict(const vector_assignment& a) const;
    
    //! implements Factor::subst_args
    nonlinear_gaussian& subst_args(const vector_var_map& map);

  private:
    //! multiplies this factor with a marginal moment Gaussian
    moment_gaussian multiply_with(const moment_gaussian& mg) const;

    friend moment_gaussian 
    operator*(const nonlinear_gaussian&, const moment_gaussian&);

    friend moment_gaussian
    operator*(const moment_gaussian&, const nonlinear_gaussian&);

  }; // class nonlinear_gaussian

  // Free functions
  //============================================================================

  //! \relates nonlinear_gaussian
  std::ostream& operator<<(std::ostream& out, const nonlinear_gaussian& cpd);

  //! linearizes nonlinear_gaussian with mg and returns the product
  //! \relates nonlinear_gaussian, moment_gaussian
  inline moment_gaussian 
  operator*(const nonlinear_gaussian& cpd, const moment_gaussian& mg) {
    return cpd.multiply_with(mg);
  }
  
  //! linearizes nonlienar_gaussian with mg and retursn the product
  //! \relates nonlinear_gaussian, moment_gaussian
  inline moment_gaussian
  operator*(const moment_gaussian& mg, const nonlinear_gaussian& cpd) {
    return cpd.multiply_with(mg);
  }

} // namespace sill

#endif
