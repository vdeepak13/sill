#ifndef SILL_PROBABILITY_TABLE_HPP
#define SILL_PROBABILITY_TABLE_HPP

#include <sill/global.hpp>
#include <sill/factor/base/table_base.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/traits.hpp>
#include <sill/functional/operators.hpp>
#include <sill/functional/entropy.hpp>
#include <sill/math/constants.hpp>

#include <initializer_list>
#include <iostream>

namespace sill {

  // Forward declaration
  template <typename T> class canonical_table;

  /**
   * A factor of a categorical probability distribution in the probability
   * space. This factor represents a non-negative function over finite
   * variables X directly using its parameters, f(X = x | \theta) = \theta_x.
   * In some cases, e.g. in a Bayesian network, this factor in fact
   * represents a (conditional) probability distribution. In other cases,
   * e.g. in a Markov network, there are no constraints on the normalization
   * of f.
   *
   * \tparam T a real type for representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T = double>
  class probability_table : public table_base<T> {
  public: 
    // Public types
    //==========================================================================

    // Factor member types
    typedef T                 real_type;
    typedef T                 result_type;
    typedef finite_variable   variable_type;
    typedef finite_domain     domain_type;
    typedef finite_var_vector var_vector_type;
    typedef finite_assignment assignment_type;
    typedef table<T>          param_type;
    
    // IndexableFactor member types
    typedef finite_index index_type;
    
    // DistributionFactor member types
    typedef boost::function<probability_table(const finite_domain&)>
      marginal_fn_type;
    typedef boost::function<probability_table(const finite_domain&,
                                              const finite_domain&)>
      conditional_fn_type;
    typedef probability_table probability_factor_type;
    
    // LearnableFactor member types
    typedef finite_dataset dataset_type;
    typedef finite_record_old record_type;

    
    // Constructors and conversion operators
    //==========================================================================

    //! Default constructor. Creates an empty factor.
    probability_table() { }

    //! Constructs a factor with given arguments and uninitialized parameters.
    explicit probability_table(const finite_var_vector& args) {
      this->reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit probability_table(T value) {
      this->reset(finite_var_vector());
      this->param_[0] = value;
    }

    //! Constructs a factor with the given arguments and constant likelihood.
    probability_table(const finite_var_vector& args, T value) {
      this->reset(args);
      this->param_.fill(value);
    }

    //! Constructs a factor with the given argument set and constant likelihood.
    probability_table(const finite_domain& args, T value) {
      this->reset(make_vector(args));
      this->param_.fill(value);
    }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const finite_var_vector& args, const table<T>& param)
      : table_base<T>(args, param) { }

    //! Creates a factor with the specified arguments and parameters.
    probability_table(const finite_var_vector& args,
                      std::initializer_list<T> values) {
      this->reset(args);
      assert(values.size() == this->size());
      std::copy(values.begin(), values.end(), this->begin());
    }

    //! Conversion from a canonical_table factor.
    explicit probability_table(const canonical_table<T>& f) {
      *this = f;
    }

    //! Assigns a constant to this factor.
    probability_table& operator=(T value) {
      this->reset(finite_var_vector());
      this->param_[0] = value;
      return *this;
    }

    //! Assigns a probability table factor to this factor.
    probability_table& operator=(const canonical_table<T>& f) {
      this->reset(f.arg_vector());
      std::transform(f.begin(), f.end(), this->begin(), exponent<T>());
      return *this;
    }

    //! Exchanges the content of two factors.
    friend void swap(probability_table& f, probability_table& g) {
      f.swap(g);
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the argument set of this factor.
    const finite_domain& arguments() const {
      return this->args_;
    }

    //! Returns the argument vector of this factor.
    const finite_var_vector& arg_vector() const {
      return this->finite_args_;
    }

    //! Returns the value of the factor for the given assignment.
    T operator()(const finite_assignment& a) const {
      return this->param(a);
    }

    //! Returns the value of the factor for the given index.
    T operator()(const finite_index& index) const {
      return this->param(index);
    }

    //! Returns the log-value of the factor for the given assignment.
    T log(const finite_assignment& a) const {
      return std::log(this->param(a));
    }

    //! Returns the log-value of the factor for the given index.
    T log(const finite_index& index) const {
      return std::log(this->param(index));
    }

    //! Returns true if the two factors have the same argument vectors and values.
    friend bool operator==(const probability_table& f, const probability_table& g) {
      return f.arg_vector() == g.arg_vector() && f.param() == g.param();
    }

    //! Returns true if the two factors do not have the same arguments or values.
    friend bool operator!=(const probability_table& f, const probability_table& g) {
      return !(f == g);
    }

    // Factor operations
    //==========================================================================

    //! Element-wise addition of two factors.
    probability_table& operator+=(const probability_table& f) {
      this->transform_inplace(f, std::plus<T>());
      return *this;
    }

    //! Element-wise subtraction of two factors.
    probability_table& operator-=(const probability_table& f) {
      this->transform_inplace(f, std::minus<T>());
      return *this;
    }
    
    //! Multiplies another factor into this one.
    probability_table& operator*=(const probability_table& f) {
      this->join_inplace(f, std::multiplies<T>());
      return *this;
    }

    //! Divides another factor into this one.
    probability_table& operator/=(const probability_table& f) {
      this->join_inplace(f, safe_divides<T>());
      return *this;
    }

    //! Increments this factor by a constant.
    probability_table& operator+=(T x) {
      this->param_.transform(incremented_by<T>(x));
      return *this;
    }

    //! Decrements this factor by a constant.
    probability_table& operator-=(T x) {
      this->param_.transform(decremented_by<T>(x));
      return *this;
    }

    //! Multiplies this factor by a constant.
    probability_table& operator*=(T x) {
      this->param_.transform(multiplied_by<T>(x));
      return *this;
    }

    //! Divides this factor by a constant.
    probability_table& operator/=(T x) {
      this->param_.transform(divided_by<T>(x));
      return *this;
    }

    //! Element-wise sum of two factors.
    friend probability_table
    operator+(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, std::plus<T>());
    }

    //! Element-wise difference of two factors.
    friend probability_table
    operator-(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, std::minus<T>());
    }

    //! Multiplies two probability_table factors.
    friend probability_table
    operator*(const probability_table& f, const probability_table& g) {
      return join<probability_table>(f, g, std::multiplies<T>());
    }

    //! Divides two probability_table factors.
    friend probability_table
    operator/(const probability_table& f, const probability_table& g) {
      return join<probability_table>(f, g, safe_divides<T>());
    }

    //! Adds a probability_table factor and a constant.
    friend probability_table
    operator+(const probability_table& f, T x) {
      return transform<probability_table>(f, incremented_by<T>(x));
    }

    //! Adds a probability_table factor and a constant.
    friend probability_table
    operator+(T x, const probability_table& f) {
      return transform<probability_table>(f, incremented_by<T>(x));
    }

    //! Subtracts a constant from a probability_table factor.
    friend probability_table
    operator-(const probability_table& f, T x) {
      return transform<probability_table>(f, decremented_by<T>(x));
    }

    //! Subtracts a probability_table factor from a constant.
    friend probability_table
    operator-(T x, const probability_table& f) {
      return transform<probability_table>(f, subtracted_from<T>(x));
    }

    //! Multiplies a probability_table factor by a constant.
    friend probability_table
    operator*(const probability_table& f, T x) {
      return transform<probability_table>(f, multiplied_by<T>(x));
    }

    //! Multiplies a probability_table factor by a constant.
    friend probability_table
    operator*(T x, const probability_table& f) {
      return transform<probability_table>(f, multiplied_by<T>(x));
    }

    //! Divides a probability_table factor by a constant.
    friend probability_table
    operator/(const probability_table& f, T x) {
      return transform<probability_table>(f, divided_by<T>(x));
    }

    //! Divides a constant by a probability_table factor.
    friend probability_table
    operator/(T x, const probability_table& f) {
      return transform<probability_table>(f, dividing<T>(x));
    }

    //! Raises the probability_table factor by an exponent.
    friend probability_table
    pow(const probability_table& f, T x) {
      return transform<probability_table>(f, exponentiated<T>(x));
    }

    //! Element-wise maximum of two factors.
    friend probability_table
    max(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, sill::maximum<T>());
    }
  
    //! Element-wise minimum of two factors.
    friend probability_table
    min(const probability_table& f, const probability_table& g) {
      return transform<probability_table>(f, g, sill::minimum<T>());
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend probability_table
    weighted_update(const probability_table& f, const probability_table& g, T a) {
      return transform<probability_table>(f, g, weighted_plus<T>(1 - a, a));
    }

    //! Computes the marginal of the factor over a subset of variables.
    probability_table marginal(const finite_domain& retain) const {
      probability_table result; marginal(retain, result);
      return result;
    }

    //! Computes the maximum for each assignment to the given variables.
    probability_table maximum(const finite_domain& retain) const {
      probability_table result; maximum(retain, result);
      return result;
    }

    //! Computes the minimum for each assignment to the given variables.
    probability_table minimum(const finite_domain& retain) const {
      probability_table result; minimum(retain, result);
      return result;
    }

    //! If this factor represents p(x, y), returns p(x | y).
    probability_table conditional(const finite_domain& tail) const {
      return (*this) / marginal(tail);
    }

    //! Computes the marginal of the factor over a subset of variables.
    void marginal(const finite_domain& retain, probability_table& result) const {
      this->aggregate(retain, T(0), std::plus<T>(), result);
    }

    //! Computes the maximum for each assignment to the given variables.
    void maximum(const finite_domain& retain, probability_table& result) const {
      this->aggregate(retain, -inf<T>(), sill::maximum<T>(), result);
    }

    //! Computes the minimum for each assignment to the given variables.
    void minimum(const finite_domain& retain, probability_table& result) const {
      this->aggregate(retain, +inf<T>(), sill::minimum<T>(), result);
    }

    //! Returns the normalization constant of the factor.
    T marginal() const {
      return this->param_.accumulate(T(0), std::plus<T>());
    }

    //! Returns the maximum value in the factor.
    T maximum() const {
      return this->param_.accumulate(-inf<T>(), sill::maximum<T>());
    }

    //! Returns the minimum value in the factor.
    T minimum() const {
      return this->param_.accumulate(+inf<T>(), sill::minimum<T>());
    }

    //! Computes the maximum value and stores the corresponding assignment.
    T maximum(finite_assignment& a) const {
      const T* it = std::max_element(this->begin(), this->end());
      this->assignment(this->param_.index(it), a);
      return *it;
    }

    //! Computes the minimum value and stores the corresponding assignment.
    T minimum(finite_assignment& a) const {
      const T* it = std::min_element(this->begin(), this->end());
      this->assignment(this->param_.index(it), a);
      return *it;
    }

    //! Normalizes the factor in-place.
    probability_table& normalize() {
      this->param_ /= marginal();
      return *this;
    }

    //! Returns true if the factor is normalizable (approximation).
    bool is_normalizable() const {
      return maximum() > 0;
    }
    
    //! Restricts this factor to an assignment.
    probability_table restrict(const finite_assignment& a) const {
      probability_table result; restrict(a, result);
      return result;
    }

    //! Restricts this factor to an assignment.
    void restrict(const finite_assignment& a, probability_table& result) const {
      table_base<T>::restrict(a, result);
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return this->param_.transform_accumulate(T(0), entropy_op<T>(), std::plus<T>());
    }

    //! Computes the entropy for a subset of variables. Performs marginalization.
    T entropy(const finite_domain& a) const {
      return (arguments() == a) ? entropy() : marginal(a).entropy();
    }

    //! Computes the mutual information between two subsets of this factor's
    //! arguments.
    T mutual_information(const finite_domain& a, const finite_domain& b) const {
      return entropy(a) + entropy(b) - entropy(set_union(a, b));
    }

    //! Computes the cross entropy from p to q.
    friend T cross_entropy(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, entropy_op<T>(), std::plus<T>());
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T kl_divergence(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, kld_op<T>(), std::plus<T>());
    }

    //! Computes the Jensen–Shannon divergece between p and q.
    friend T js_divergence(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, jsd_op<T>(), std::plus<T>());
    }

    //! Computes the sum of absolute differences between the parameters of p and q.
    friend T sum_diff(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), std::plus<T>());
    }
    
    //! Computes the max of absolute differences between the parameters of p and q.
    friend T max_diff(const probability_table& p, const probability_table& q) {
      return transform_accumulate(p, q, abs_difference<T>(), sill::maximum<T>());
    }

    /**
     * A type that represents the log-likelihood function and its derivatives.
     * Models the LogLikelihoodObjective concept.
     */
    struct loglikelihood_type {
      const table<T>& f;
      loglikelihood_type(const table<T>* f) : f(*f) { }
      
      void add_gradient(const finite_index& index, T w, table<T>& g) {
        g(index) += w / f(index);
      }

      void add_gradient(const table<T>& phead, const finite_index& tail, T w,
                        table<T>& g) {
        assert(phead.arity() + tail.size() == g.arity());
        size_t index = g.offset().linear(tail, phead.arity());
        for (size_t i = 0; i < phead.size(); ++i) {
          g[index + i] += phead[i] * w / f[index + i];
        }
      }
      
      void add_gradient_sqr(const table<T>& phead, const finite_index& tail, T w,
                            table<T>& g) {
        add_gradient(phead, tail, w, g);
      }

      void add_hessian_diag(const finite_index& index, T w, table<T>& h) {
        T fval = f(index);
        h(index) -= w / (fval * fval);
      }

      void add_hessian_diag(const table<T>& phead, const finite_index& tail, T w,
                            table<T>& h) {
        assert(phead.arity() + tail.size() == h.arity());
        size_t index = h.offset().linear(tail, phead.arity());
        for (size_t i = 0; i < phead.size(); ++i) {
          h[index + i] -= phead[i] * w / (f[index + i] * f[index + i]);
        }
      }
    }; // struct loglikelihood_type

  }; // class probability_table

  // Input / output
  //============================================================================

  /**
   * Prints a human-readable representatino of the table factor to the stream.
   * \relates probability_table
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const probability_table<T>& f) {
    out << "#PT(" << f.arg_vector() << ")" << std::endl;
    out << f.param();
    return out;
  }

  // Utilities - TODO
  //============================================================================


  // Traits
  //============================================================================

  template <typename T>
  struct has_multiplies<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_multiplies_assign<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_divides<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_divides_assign<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_max<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_min<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_marginal<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_maximum<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_minimum<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_arg_max<probability_table<T> > : public boost::true_type { };

  template <typename T>
  struct has_arg_min<probability_table<T> > : public boost::true_type { };

} // namespace sill

#endif
