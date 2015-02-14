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
  template <typename T> class canonical_array;
  template <typename T> class probability_table;

  /**
   * A factor of a categorical probability distribution that may contain
   * up to 2 arguments in its domain. This factor represents a non-negative
   * function over finite variables X, Y with a parameter array \theta as
   * f(X = x, Y = y | \theta) = \theta_{x,y}, and similarly for unary and
   * nullary functions. In some cases, this class represents a array of
   * probabilities (e.g., when used in a Bayesian network). In other cases,
   * e.g. in a pairwise Markov network, there are no constraints on the
   * normalization of f.
   * 
   * \tparam T a type of values stored in the factor
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T = double>
  class probability_array : public array_factor<T> {
  public:
    // Factor member types
    typedef T                                    real_type;
    typedef T                                    result_type;
    typedef finite_variable                      variable_type;
    typedef bounded_domain<finite_variable*, 2>  domain_type;
    typedef bounded_domain<finite_variable*, 2>  var_vector_type;
    typedef finite_assignment                    assignment_type;
    typedef typename array_factor<T>::array_type param_type;
    
    // IndexableFactor member types
    typedef finite_index index_type;
    
    // DistributionFactor member types
    typedef boost::function<probability_array(const domain_type&)>
      marginal_fn_type;
    typedef boost::function<probability_array(const domain_type&,
                                              const domain_type&)>
      conditional_fn_type;
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

    //! Constructs a factor equivalent to a constant.
    explicit probability_array(T value) {
      this->reset();
      this->param_(0) = value;
    }

    //! Constructs a factor with the given arguments and constant value.
    probability_array(const domain_type& args, T value) {
      this->reset(args);
      this->param_.fill(value);
    }

    //! Constructs a factor with the given argument and parameters.
    probability_array(const domain_type& args,
                      const param_type& param,
                      bool zero_nan = false)
      : array_factor<T>(args, param, zero_nan) { }

    probability_array(const domain_type& args,
                      param_type&& param,
                      bool zero_nan = false)
      : array_factor<T>(args, std::move(param), zero_nan) { }

    //! Constructs a factor with the given arguments and parameters.
    probability_array(const domain_type& args,
                      std::initializer_list<T> values) {
      this->reset(args);
      assert(this->size() == values.size());
      std::copy(values.begin(), values.end(), this->begin());
    }

    //! Conversion from a canonical_array factor.
    explicit probability_array(const canonical_array<T>& f) {
      *this = f;
    }

    //! Conversion from a probability_table factor.
    explicit probability_array(const probability_table<T>& f) {
      *this = f;
    }

    //! Assigns a constant to this factor.
    probability_array& operator=(T value) {
      this->reset();
      this->param_(0) = value;
      return *this;
    }

    //! Assigns a canonical_array factor to this factor.
    probability_array& operator=(const canonical_array<T>& f) {
      this->reset(f.arguments());
      this->param_ = exp(f.param());
      return *this;
    }
    
    //! Assigns a probability_table to this factor
    probability_array& operator=(const probability_table<T>& f) {
      this->reset(domain_type(f.arg_vector()));
      assert(this->size() == f.size());
      std::copy(f.begin(), f.end(), this->begin());
      return *this;
    }

    //! Swaps the content of two probability_array factors.
    friend void swap(probability_array& f, probability_array& g) {
      f.swap(g);
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
    
    //! Multiplies another factor into this one.
    probability_array& operator*=(const probability_array& f) {
      this->join_inplace(f, sill::multiplies_assign<>(), false);
      return *this;
    }

    //! Divides another factor into this one.
    probability_array& operator/=(const probability_array& f) {
      this->join_inplace(f, sill::divides_assign<>(), true /* support 0/0 */);
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

    //! Multiplies two probability_array factors.
    friend probability_array
    operator*(const probability_array& f, const probability_array& g) {
      return join<probability_array>(f, g, sill::multiplies<>());
    }

    //! Divides two probability_array factors.
    friend probability_array
    operator/(const probability_array& f, const probability_array& g) {
      return join<probability_array>(f, g, sill::divides<>(), true /* 0/0 */);
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

    //! Computes the marginal of the factor over a subset of variables.
    probability_array marginal(const domain_type& retain) const {
      probability_array result; marginal(retain, result);
      return result;
    }

    //! Computes the maximum for each assignment to the given variables.
    probability_array maximum(const domain_type& retain) const {
      probability_array result; maximum(retain, result);
      return result;
    }

    //! Computes the minimum for each assignment to the given variables.
    probability_array minimum(const domain_type& retain) const {
      probability_array result; minimum(retain, result);
      return result;
    }

    //! If this factor represents p(x, y), returns p(x | y).
    probability_array conditional(const domain_type& tail) const {
      return (*this) / marginal(tail);
    }

    //! Computes the marginal of the factor over a subset of variables.
    void marginal(const domain_type& retain, probability_array& result) const {
      this->aggregate(retain, sum_op(), result);
    }

    //! Computes the maximum for each assignment to the given variables.
    void maximum(const domain_type& retain, probability_array& result) const {
      this->aggregate(retain, max_coeff_op(), result);
    }

    //! Computes the minimum for each assignment to the given variables.
    void minimum(const domain_type& retain, probability_array& result) const {
      this->aggregate(retain, min_coeff_op(), result);
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
    
    //! Restricts this factor to an assignment.
    probability_array restrict(const finite_assignment& a) const {
      probability_array result; restrict(a, result);
      return result;
    }

    //! Restricts this factor to an assignment.
    void restrict(const finite_assignment& a, probability_array& result) const {
      array_factor<T>::restrict(a, result);
    }

    //! Restricts this factor to an assignment and multiplies it into the result.
    void restrict_multiply(const finite_assignment& a,
                           probability_array& result) const {
      array_factor<T>::restrict_join(a, multiplies_assign<>(), false, result);
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return this->transform_accumulate(T(0), entropy_op<T>(), std::plus<T>());
    }

    //! Computes the entropy for a subset of variables. Performs marginalization.
    T entropy(const domain_type& a) const {
      return equivalent(this->arguments(), a) ? entropy() : marginal(a).entropy();
    }

    //! Computes the mutual information between two subsets of this factor's
    //! arguments.
    T mutual_information(const domain_type& a, const domain_type& b) const {
      return entropy(a) + entropy(b) - entropy(left_union(a, b));
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
      const param_type& a;
      loglikelihood_type(const param_type* a) : a(*a) { }
      
      void add_gradient(const finite_index& index, T w, param_type& g) {
        check_shape_compatible(g);
        size_t offset = linear_index(index);
        g(offset) += w / a(offset);
      }

      void add_gradient(const param_type& phead, const finite_index& tail, T w,
                        param_type& g) {
        check_shape_compatible(g);
        size_t offset = linear_index(tail, phead.size());
        size_t size = phead.size();
        for (size_t i = 0; i < size; ++i) {
          g(offset + i) += phead(i) * w / a(offset + i);
        }
      }
      
      void add_gradient_sqr(const param_type& phead, const finite_index& tail, T w,
                            param_type& g) {
        add_gradient(phead, tail, w, g);
      }

      void add_hessian_diag(const finite_index& index, T w, param_type& h) {
        check_shape_compatible(h);
        size_t offset = linear_index(index);
        h(offset) -= w / (a(offset) * a(offset));
      }

      void add_hessian_diag(const param_type& phead, const finite_index& tail, T w,
                            param_type& h) {
        check_shape_compatible(h);
        size_t offset = linear_index(tail, phead.size());
        size_t size = phead.size();
        for (size_t i = 0; i < size; ++i) {
          h(offset + i) -= phead(i) * w / (a(offset + i) * a(offset + i));
        }
      }

      size_t linear_index(const finite_index& index, size_t mult = 1) const {
        switch (index.size()) {
        case 0:
          assert(mult == a.size());
          return 0;
        case 1:
          assert(mult == 1 && a.cols() == 1 || mult == a.rows());
          return index[0] * mult;
        case 2:
          assert(mult == 1);
          return index[0] + index[1] * a.rows();
        default:
          throw std::invalid_argument(
            "An index with >2 elements passed to an array factor"
          );
        }
      }

      void check_shape_compatible(const param_type& x) const {
        if (a.rows() != x.rows() || a.cols() != x.cols()) {
          throw std::invalid_argument("Incompatible shape");
        }
      }

    }; // struct loglikelihood_type

  }; // class probability_array

  /**
   * Outputs a human-readable representation of ths factor to the stream.
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const probability_array<T>& f) {
    out << f.arg_vector() << std::endl
        << f.param() << std::endl;
    return out;
  }

} // namespace sill

#endif
