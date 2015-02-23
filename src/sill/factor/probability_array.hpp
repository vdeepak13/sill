#ifndef SILL_PROBABILITY_ARRAY_HPP
#define SILL_PROBABILITY_ARRAY_HPP

#include <sill/global.hpp>
#include <sill/functional.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/factor/base/array_factor.hpp>
#include <sill/functional/assign.hpp>
#include <sill/functional/eigen.hpp>
#include <sill/functional/entropy.hpp>
#include <sill/learning/dataset/finite_dataset.hpp>

#include <armadillo>
#include <boost/function.hpp>

#include <iostream>

namespace sill {

  // Forward declarations
  template <typename T, size_t N> class canonical_array;
  template <typename T> class probability_table;

  /**
   * A factor of a categorical probability distribution that contains
   * one or two arguments in its domain, where the number of arguments is
   * fixed at compile-time. The factor represents a non-negative function
   * directly with a parameter array \theta as
   * f(X = x, Y = y | \theta) = \theta_{x,y} for binary factors and
   * f(X = x | \theta) = \theta_x for unary factors.
   * In some cases, this class represents a array of probabilities
   * (e.g., when used in a hidden Markov model). In other cases, e.g.
   * in a pairwise Markov network, there are no constraints on the
   * normalization of f.
   * 
   * \tparam T a type of values stored in the factor
   * \tparam N the number of arguments
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T, size_t N>
  class probability_array : public array_factor<T, N> {
  public:
    // Helper types
    typedef array_factor<T, N> base;
    typedef array_domain<finite_variable*, 1> unary_domain_type;
 
    // Factor member types
    typedef T                                 real_type;
    typedef T                                 result_type;
    typedef finite_variable                   variable_type;
    typedef array_domain<finite_variable*, N> domain_type;
    typedef array_domain<finite_variable*, N> var_vector_type;
    typedef finite_assignment                 assignment_type;
    typedef typename base::array_type         param_type;
    
    // IndexableFactor member types
    typedef finite_index index_type;
    
    // DistributionFactor member types
    typedef probability_array probability_factor_type;
    
    // LearnableFactor types
    typedef finite_dataset dataset_type;
    typedef finite_record  record_type;

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Default constructor. Creates an empty factor.
    probability_array() { }

    //! Constructs a factor with the given arguments and uninitialized parameters.
    explicit probability_array(const domain_type& args) {
      this->reset(args);
    }

    //! Constructs a factor with the given arguments and constant value.
    probability_array(const domain_type& args, T value) {
      this->reset(args);
      this->param_.fill(value);
    }

    //! Constructs a factor with the given argument and parameters.
    probability_array(const domain_type& args, const param_type& param)
      : base(args, param) { }

    //! Constructs a factor with the given argument and parameters.
    probability_array(const domain_type& args, param_type&& param)
      : base(args, std::move(param)) { }

    //! Constructs a factor with the given arguments and parameters.
    probability_array(const domain_type& args, std::initializer_list<T> values)
      : base(args, values) { }

    //! Conversion from a canonical_array factor.
    explicit probability_array(const canonical_array<T, N>& f) {
      *this = f;
    }

    //! Conversion from a probability_table factor.
    explicit probability_array(const probability_table<T>& f) {
      *this = f;
    }

    //! Assigns a canonical_array factor to this factor.
    probability_array& operator=(const canonical_array<T, N>& f) {
      this->reset(f.arguments());
      this->param_ = exp(f.param());
      return *this;
    }
    
    //! Assigns a probability_table to this factor
    probability_array& operator=(const probability_table<T>& f) {
      this->reset(f.arg_vector());
      assert(f.size() == this->size());
      std::copy(f.begin(), f.end(), this->begin());
      return *this;
    }

    //! Swaps the content of two probability_array factors.
    friend void swap(probability_array& f, probability_array& g) {
      f.swap(g);
    }

    //! Replaces NaNs with 0s (used when dividing two probability_array factors).
    probability_array& clear_nan() {
      for (T& value : *this) {
        if (std::isnan(value)) { value = T(0); }
      }
      return *this;
    }
    
    // Accessors
    //==========================================================================
    //! Returns the argument vector. Deprecated.
    const var_vector_type& arg_vector() const {
      return this->args_;
    }
    
    //! Returns the value of this factor for an assignment
    T operator()(const finite_assignment& a) const {
      return this->param(a);
    }

    //! Returns the value of this factor for an index
    T operator()(const finite_index& index) const {
      return this->param(index);
    }

    //! Returns the log-value of the factor for the given assignment.
    T log(const finite_assignment& a) const {
      return std::log(this->param(a));
    }

    //! Returns the log-value of teh factor for the given index.
    T log(const finite_index& index) const {
      return std::log(this->param(index));
    }

    //! Returns true if the two factors have the same argument vectors and values.
    friend bool operator==(const probability_array& f,
                           const probability_array& g) {
      return f->equal(g);
    }

    //! Returns true if the two factors do not have the same arguments or values.
    friend bool operator!=(const probability_array& f,
                           const probability_array& g) {
      return !f->equal(g);
    }

    // Factor operations
    //==========================================================================

    //! Element-wise addition of two factors.
    probability_array& operator+=(const probability_array& f) {
      check_same_arguments(*this, f);
      this->param_ += f.param_;
      return *this;
    }

    //! Element-wise subtraction of two factors.
    probability_array& operator-=(const probability_array& f) {
      check_same_arguments(*this, f);
      this->param_ -= f.param_;
      return *this;
    }
    
    /**
     * Multiplies another factor with arity M into this one.
     * This operation is only supported when M <= N, i.e.,
     * the given factor has no more arguments than this one.
     */
    template <size_t M>
    typename std::enable_if<M <= N, probability_array&>::type
    operator*=(const probability_array<T, M>& f) {
      join_inplace(*this, f, sill::multiplies_assign<>());
      return *this;
    }

    /**
     * Divides another factor with arity M into this one.
     * This operation is only supported when M <= N, i.e.,
     * the given factor has no more arguments than this one.
     */
    template <size_t M>
    typename std::enable_if<M <= N, probability_array&>::type
    operator/=(const probability_array<T, M>& f) {
      join_inplace(*this, f, sill::divides_assign<>());
      clear_nan();
      return *this;
    }

    //! Increments this factor by a constant.
    probability_array& operator+=(T x) {
      this->param_ += x;
      return *this;
    }

    //! Decrements this factor by a constant.
    probability_array& operator-=(T x) {
      this->param_ -= x;
      return *this;
    }

    //! Multiplies this factor by a constant.
    probability_array& operator*=(T x) {
      this->param_ *= x;
      return *this;
    }

    //! Divides this factor by a constant.
    probability_array& operator/=(T x) {
      this->param_ /= x;
      return *this;
    }

    //! Element-wise sum of two factors.
    friend probability_array
    operator+(const probability_array& f, const probability_array& g) {
      check_same_arguments(f, g);
      return probability_array(f.arguments(), f.param() + g.param());
    }

    //! Element-wise difference of two factors.
    friend probability_array
    operator-(const probability_array& f, const probability_array& g) {
      check_same_arguments(f, g);
      return probability_array(f.arguments(), f.param() - g.param());
    }

    //! Adds a probability_array factor and a constant.
    friend probability_array
    operator+(const probability_array& f, T x) {
      return probability_array(f.arguments(), f.param() + x);
    }

    //! Adds a probability_array factor and a constant.
    friend probability_array
    operator+(T x, const probability_array& f) {
      return probability_array(f.arguments(), x + f.param());
    }

    //! Subtracts a constant from a probability_array factor.
    friend probability_array
    operator-(const probability_array& f, T x) {
      return probability_array(f.arguments(), f.param() - x);
    }

    //! Subtracts a probability_array factor from a constant.
    friend probability_array
    operator-(T x, const probability_array& f) {
      return probability_array(f.arguments(), x - f.param());
    }

    //! Multiplies a probability_array factor by a constant.
    friend probability_array
    operator*(const probability_array& f, T x) {
      return probability_array(f.arguments(), f.param() * x);
    }

    //! Multiplies a probability_array factor by a constant.
    friend probability_array
    operator*(T x, const probability_array& f) {
      return probability_array(f.arguments(), x * f.param());
    }

    //! Divides a probability_array factor by a constant.
    friend probability_array
    operator/(const probability_array& f, T x) {
      return probability_array(f.arguments(), f.param() / x);
    }

    //! Divides a constant by a probability_array factor.
    friend probability_array
    operator/(T x, const probability_array& f) {
      return probability_array(f.arguments(), x / f.param());
    }

    //! Raises the probability_array factor by an exponent.
    friend probability_array
    pow(const probability_array& f, T x) {
      return probability_array(f.arguments(), f.param().pow(x));
    }

    //! Element-wise maximum of two factors.
    friend probability_array
    max(const probability_array& f, const probability_array& g) {
      check_same_arguments(f, g);
      return probability_array(f.arguments(), f.param().max(g.param()));
    }
  
    //! Element-wise minimum of two factors.
    friend probability_array
    min(const probability_array& f, const probability_array& g) {
      check_same_arguments(f, g);
      return probability_array(f.arguments(), f.param().min(g.param()));
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend probability_array
    weighted_update(const probability_array& f,
                    const probability_array& g, T a) {
      check_same_arguments(f, g);
      return probability_array(f.arguments(),
                               (1-a) * f.param() + a * g.param());
    }

    /**
     * Computes the marginal of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, probability_array<T, 1> >::type
    marginal(const unary_domain_type& retain) const {
      return aggregate<probability_array<T, 1>>(*this, retain, sum_op());
    }
    
    /**
     * Computes the maximum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, probability_array<T, 1> >::type
    maximum(const unary_domain_type& retain) const {
      return aggregate<probability_array<T, 1>>(*this, retain, max_coeff_op());
    }

    /**
     * Computes the minimum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, probability_array<T, 1> >::type
    minimum(const unary_domain_type& retain) const {
      return aggregate<probability_array<T, 1>>(*this, retain, min_coeff_op());
    }

    /**
     * If this factor represents p(x, y) where y = tail, returns p(x | y).
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, probability_array>::type
    conditional(const unary_domain_type& tail) const {
      return (*this) / marginal(tail);
    }

    /**
     * Computes the marginal of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    marginal(const unary_domain_type& retain,
             probability_array<T, 1>& result) const {
      aggregate(*this, retain, result, sum_op());
    }

    /**
     * Computes the maximum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    maximum(const unary_domain_type& retain,
            probability_array<T, 1>& result) const {
      aggregate(*this, retain, result, max_coeff_op());
    }

    /**
     * Computes the minimum of the factor over a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    minimum(const unary_domain_type& retain,
            probability_array<T, 1>& result) const {
      aggregate(*this, retain, result, min_coeff_op());
    }

    //! Returns the normalization constant of the factor.
    T marginal() const {
      return this->param_.sum();
    }

    //! Returns the maximum value in the factor.
    T maximum() const {
      return this->param_.maxCoeff();
    }

    //! Returns the minimum value in the factor.
    T minimum() const {
      return this->param_.minCoeff();
    }

    //! Computes the maximum value and stores the corresponding assignment.
    T maximum(finite_assignment& a) const {
      const T* it = std::max_element(this->begin(), this->end());
      this->assignment(it - this->begin(), a);
      return *it;
    }

    //! Computes the minimum value and stores the corresponding assignment.
    T minimum(finite_assignment& a) const {
      const T* it = std::min_element(this->begin(), this->end());
      this->assignment(it - this->begin(), a);
      return *it;
    }

    //! Normalizes the factor in-place.
    probability_array& normalize() {
      this->param_ /= marginal();
      return *this;
    }

    //! Returns true if the factor is normalizable (approximation).
    bool is_normalizable() const {
      return maximum() > 0;
    }

    /**
     * Restricts the factor to an assignment and returns the result
     * as a unary factor. This operation is only supported for binary
     * factors, and the assignment must restrict exactly one argument.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, probability_array<T, 1> >::type
    restrict(const finite_assignment& a) const {
      probability_array<T, 1> result;
      restrict_assign(*this, a, result);
      return result;
    }
    
    /**
     * Restricts this factor to an assignment and stores the result
     * in a unary factor. This operation is only supported for binary
     * factors, and teh assignment must restrict exactly one argument.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    restrict(const finite_assignment& a,
             probability_array<T, 1>& result) const {
      restrict_assign(*this, a, result);
    }

    /**
     * Restricts this factor to an assignment, excluding the variables in
     * the unary factor result, and multiplies the restriction into result.
     * This operation must not introduce any new variables and is only
     * supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B>::type
    restrict_multiply(const finite_assignment& a,
                      probability_array<T, 1>& result) const {
      restrict_join(*this, a, result, multiplies_assign<>());
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return transform_accumulate(*this, entropy_op<T>(), std::plus<T>());
    }

    /**
     * Computes the entropy for a single variable.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, T>::type
    entropy(const unary_domain_type& a) const {
      return marginal(a).entropy();
    }

    /**
     * Computes the mutual information between two variables.
     * This operation is only supported for binary factors.
     */
    template <bool B = (N == 2)>
    typename std::enable_if<B, T>::type
    mutual_information(const unary_domain_type& a,
                       const unary_domain_type& b) const {
      assert(subset(a, this->arguments()));
      assert(subset(b, this->arguments()));
      return entropy(a) + entropy(b) - entropy();
    }

    //! Computes the cross entropy from p to q.
    friend T cross_entropy(const probability_array& p,
                           const probability_array& q) {
      return transform_accumulate(p, q, entropy_op<T>(), std::plus<T>());
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T kl_divergence(const probability_array& p,
                           const probability_array& q) {
      return transform_accumulate(p, q, kld_op<T>(), std::plus<T>());
    }

    //! Computes the Jensen–Shannon divergece between p and q.
    friend T js_divergence(const probability_array& p,
                           const probability_array& q) {
      return transform_accumulate(p, q, jsd_op<T>(), std::plus<T>());
    }

    //! Computes the sum of absolute differences between the parameters of p and q.
    friend T sum_diff(const probability_array& p,
                      const probability_array& q) {
      return transform_accumulate(p, q, abs_difference<T>(), std::plus<T>());
    }
    
    //! Computes the max of absolute differences between the parameters of p and q.
    friend T max_diff(const probability_array& p,
                      const probability_array& q) {
      return transform_accumulate(p, q, abs_difference<T>(), sill::maximum<T>());
    }

    /**
     * A type that represents the log-likelihood function and its derivatives.
     * Models the LogLikelihoodObjective concept.
     */
    struct loglikelihood_type {
      typedef Eigen::Array<T, Eigen::Dynamic, 1> array1_type;
      const param_type& a;
      loglikelihood_type(const param_type* a) : a(*a) { }
      
      void add_gradient(size_t i, T w, param_type& g) {
        g(i) += w / a(i);
      }

      void add_gradient(size_t i, size_t j, T w, param_type& g) {
        g(i, j) += w / a(i, j);
      }

      void add_gradient(const array1_type& phead, size_t j, T w,
                        param_type& g) {
        g.col(j) += w * phead / a.col(j);
      }
      
      void add_hessian_diag(size_t i, T w, param_type& h) {
        h(i) -= w / (a(i) * a(i));
      }

      void add_hessian_diag(size_t i, size_t j, T w, param_type& h) {
        h(i, j) -= w / (a(i, j) * a(i, j));
      }

      void add_hessian_diag(const array1_type& phead, size_t j, T w,
                            param_type& h) {
        h.col(j) -= w * phead / a.col(j) / a.col(j);
      }

    }; // struct loglikelihood_type

  }; // class probability_array

  /**
   * Outputs a human-readable representation of the factor to the stream.
   * \relates probability_array
   */
  template <typename T, size_t N>
  std::ostream& operator<<(std::ostream& out, const probability_array<T, N>& f) {
    out << f.arguments() << std::endl
        << f.param() << std::endl;
    return out;
  }

  /**
   * Multiplies two probability_array factors.
   * \tparam M the arity of the first argument
   * \tparam N the arity of the second argument
   * \return a probability_array factor whose arity is the maximum of M and N
   * \relates probability_array
   */
  template <typename T, size_t M, size_t N>
  probability_array<T, (M >= N) ? M : N>
  operator*(const probability_array<T, M>& f,const probability_array<T, N>& g) {
    typedef probability_array<T, (M >= N) ? M : N> result_type;
    return join<result_type>(f, g, sill::multiplies<>());
  }

  /**
   * Divides two probability_array factors.
   * \tparam M the arity of the first argument
   * \tparam N the arity of the second argument
   * \return a probability_array factor whose arity is the maximum of M and N
   * \relates probability_array
   */
  template <typename T, size_t M, size_t N>
  probability_array<T, (M >= N) ? M : N>
  operator/(const probability_array<T, M>& f,const probability_array<T, N>& g) {
    typedef probability_array<T, (M >= N) ? M : N> result_type;
    return join<result_type>(f, g, sill::divides<>()).clear_nan();
  }

  /**
   * Multiplies a binary and a unary probability_array factor
   * and computes the marginal over a single variable.
   */
  template <typename T>
  probability_array<T, 1>
  product_marginal(const probability_array<T, 2>& f,
                   const probability_array<T, 1>& g,
                   const array_domain<finite_variable*, 1>& retain) {
    assert(f.arguments().count(g.x()));
    assert(f.arguments().count(retain[0]));
    if (g.x() == retain[0]) {
      return f.marginal(retain) *= g;
    } else {
      return expectation<probability_array<T, 1> >(f, g);
    }
  }

  /**
   * Multiplies a unary and a binary probability_array factor
   * and computes the marginal over a single variable.
   */
  template <typename T>
  probability_array<T, 1>
  product_marginal(const probability_array<T, 1>& g,
                   const probability_array<T, 2>& f,
                   const array_domain<finite_variable*, 1>& retain) {
    assert(f.arguments().count(g.x()));
    assert(f.arguments().count(retain[0]));
    if (g.x() == retain[0]) {
      return f.marginal(retain) *= g;
    } else {
      return expectation<probability_array<T, 1> >(f, g);
    }
  }
 
} // namespace sill

#endif
