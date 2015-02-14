#ifndef SILL_CANONICAL_ARRAY_HPP
#define SILL_CANONICAL_ARRAY_HPP

#include <sill/global.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/factor/base/array_factor.hpp>
#include <sill/functional/assign.hpp>
#include <sill/functional/eigen.hpp>
#include <sill/functional/entropy.hpp>
#include <sill/functional/operators.hpp>
#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/math/constants.hpp>
#include <sill/math/logarithmic.hpp>

#include <armadillo>
#include <boost/function.hpp>

#include <iostream>

namespace sill {

  // Forward declarations
  template <typename T> class canonical_table;
  template <typename T> class probability_array;

  /**
   * A factor of a categorical canonical distribution that may contain
   * up to 2 arguments in its domain. This factor represents a non-negative
   * function over finite variables X, Y using canonical (natural) parameters
   * of the exponential family as f(X = x, Y = y | \theta) = exp(\theta_{x,y}),
   * and similarly for unary and nullary functions. In some cases, e.g., when
   * used in a Bayesian network, this factor also represents a probability
   * distribution in the log-space.
   * 
   * \tparam T a real type that represents each parameter
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T = double>
  class canonical_array : public array_factor<T> {
  public:
    // Factor member types
    typedef T                                    real_type;
    typedef logarithmic<T>                       result_type;
    typedef finite_variable                      variable_type;
    typedef bounded_domain<finite_variable*, 2>  domain_type;
    typedef bounded_domain<finite_variable*, 2>  var_vector_type;
    typedef finite_assignment                    assignment_type;
    typedef typename array_factor<T>::array_type param_type;
    
    // IndexableFactor member types
    typedef finite_index index_type;
    
    // DistributionFactor member types
    typedef boost::function<canonical_array(const domain_type&)>
      marginal_fn_type;
    typedef boost::function<canonical_array(const domain_type&,
                                            const domain_type&)>
      conditional_fn_type;
    typedef probability_array<T> probability_factor_type;
    
    // LearnableFactor types
    typedef finite_dataset dataset_type;
    typedef finite_record  record_type;

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Default constructor. Creates an empty factor.
    canonical_array() { }

    //! Constructs a factor with the given arguments and uninitialized parameters.
    explicit canonical_array(const domain_type& args) {
      this->reset(args);
    }

    //! Constructs a factor equivalent to a constant.
    explicit canonical_array(logarithmic<T> value) {
      this->reset();
      this->param_(0) = value.lv;
    }

    //! Constructs a factor with the given arguments and constant value.
    canonical_array(const domain_type& args, logarithmic<T> value) {
      this->reset(args);
      this->param_.fill(value.lv);
    }

    //! Constructs a factor with the given argument and parameters.
    canonical_array(const domain_type& args,
                    const param_type& param,
                    bool zero_nan = false)
      : array_factor<T>(args, param, zero_nan) { }

    canonical_array(const domain_type& args,
                    param_type&& param,
                    bool zero_nan = false)
      : array_factor<T>(args, std::move(param), zero_nan) { }

    //! Constructs a factor with the given arguments and parameters.
    canonical_array(const domain_type& args,
                    std::initializer_list<T> values) {
      this->reset(args);
      assert(this->size() == values.size());
      std::copy(values.begin(), values.end(), this->begin());
    }

    //! Conversion from a probability_array factor.
    explicit canonical_array(const probability_array<T>& f) {
      *this = f;
    }

    //! Conversion from a canonical_table factor.
    explicit canonical_array(const canonical_table<T>& f) {
      *this = f;
    }

    //! Assigns a constant to this factor.
    canonical_array& operator=(logarithmic<T> value) {
      this->reset();
      this->param_(0) = value.lv;
      return *this;
    }

    //! Assigns a probability_array factor to this factor.
    canonical_array& operator=(const probability_array<T>& f) {
      this->reset(f.arguments());
      this->param_ = Eigen::log(f.param());
      return *this;
    }
    
    //! Assigns a canonical_table to this factor
    canonical_array& operator=(const canonical_table<T>& f) {
      this->reset(f.arg_vector());
      assert(this->size() == f.size());
      std::copy(f.begin(), f.end(), this->begin());
      return *this;
    }

    //! Swaps the content of two canonical_array factors.
    friend void swap(canonical_array& f, canonical_array& g) {
      f.swap(g);
    }

    // Accessors
    //==========================================================================
    //! Returns the argument vector.
    const var_vector_type& arg_vector() const {
      return this->args_;
    }
    
    //! Returns the value of this factor for an assignment
    logarithmic<T> operator()(const finite_assignment& a) const {
      return logarithmic<T>(this->param(a), log_tag());
    }

    //! Returns the value of this factor for an index
    logarithmic<T> operator()(const finite_index& index) const {
      return logarithmic<T>(this->param(index), log_tag());
    }

    //! Returns the log-value of the factor for the given assignment.
    T log(const finite_assignment& a) const {
      return this->param(a);
    }

    //! Returns the log-value of teh factor for the given index.
    T log(const finite_index& index) const {
      return this->param(index);
    }

    //! Returns true if the two factors have the same argument vectors and values.
    friend bool operator==(const canonical_array& f,
                           const canonical_array& g) {
      return f->equal(g);
    }

    //! Returns true if the two factors do not have the same arguments or values.
    friend bool operator!=(const canonical_array& f,
                           const canonical_array& g) {
      return !f->equal(g);
    }

    // Factor operations
    //==========================================================================

    //! Multiplies another factor into this one.
    canonical_array& operator*=(const canonical_array& f) {
      this->join_inplace(f, sill::plus_assign<>(), false);
      return *this;
    }

    //! Divides another factor into this one.
    canonical_array& operator/=(const canonical_array& f) {
      this->join_inplace(f, sill::minus_assign<>(), false);
      return *this;
    }

    //! Multiplies this factor by a constant.
    canonical_array& operator*=(logarithmic<T> x) {
      this->param_ += x.lv;
      return *this;
    }

    //! Divides this factor by a constant.
    canonical_array& operator/=(logarithmic<T> x) {
      this->param_ -= x.lv;
      return *this;
    }

    //! Returns the sum of hte probabilities of two factors.
    friend canonical_array
    operator+(const canonical_array& f, const canonical_array& g) {
      return transform<canonical_array>(f, g, log_sum_exp<T>());
    }

    //! Multiplies two canonical_array factors.
    friend canonical_array
    operator*(const canonical_array& f, const canonical_array& g) {
      return join<canonical_array>(f, g, sill::plus<>());
    }

    //! Divides two canonical_array factors.
    friend canonical_array
    operator/(const canonical_array& f, const canonical_array& g) {
      return join<canonical_array>(f, g, sill::minus<>());
    }

    //! Multiplies a canonical_array factor by a constant.
    friend canonical_array
    operator*(const canonical_array& f, logarithmic<T> x) {
      return canonical_array(f.arguments(), f.param() + x.lv);
    }

    //! Multiplies a canonical_array factor by a constant.
    friend canonical_array
    operator*(logarithmic<T> x, const canonical_array& f) {
      return canonical_array(f.arguments(), x.lv + f.param());
    }

    //! Divides a canonical_array factor by a constant.
    friend canonical_array
    operator/(const canonical_array& f, logarithmic<T> x) {
      return canonical_array(f.arguments(), f.param() - x.lv);
    }

    //! Divides a constant by a canonical_array factor.
    friend canonical_array
    operator/(logarithmic<T> x, const canonical_array& f) {
      return canonical_array(f.arguments(), x.lv - f.param());
    }

    //! Raises the canonical_array factor by an exponent.
    friend canonical_array
    pow(const canonical_array& f, T x) {
      return canonical_array(f.arguments(), f.param() * x);
    }

    //! Element-wise maximum of two factors.
    friend canonical_array
    max(const canonical_array& f, const canonical_array& g) {
      check_same_arguments(f, g);
      return canonical_array(f.arguments(), f.param().max(g.param()));
    }
  
    //! Element-wise minimum of two factors.
    friend canonical_array
    min(const canonical_array& f, const canonical_array& g) {
      check_same_arguments(f, g);
      return canonical_array(f.arguments(), f.param().min(g.param()));
    }

    //! Returns \f$f^{(1-a)} * g^a\f$.
    friend canonical_array
    weighted_update(const canonical_array& f,
                    const canonical_array& g, T a) {
      check_same_arguments(f, g);
      return canonical_array(f.arguments(),
                             (1-a) * f.param() + a * g.param());
    }

    //! Computes the marginal of the factor over a subset of variables.
    canonical_array marginal(const domain_type& retain) const {
      canonical_array result; marginal(retain, result);
      return result;
    }

    //! Computes the maximum for each assignment to the given variables.
    canonical_array maximum(const domain_type& retain) const {
      canonical_array result; maximum(retain, result);
      return result;
    }

    //! Computes the minimum for each assignment to the given variables.
    canonical_array minimum(const domain_type& retain) const {
      canonical_array result; minimum(retain, result);
      return result;
    }

    //! If this factor represents p(x, y), returns p(x | y).
    canonical_array conditional(const domain_type& tail) const {
      return (*this) / marginal(tail);
    }

    //! Computes the marginal of the factor over a subset of variables.
    void marginal(const domain_type& retain, canonical_array& result) const {
      if (retain.empty()) {
        result = canonical_array(marginal());
      } else {
        T max = this->param_.maxCoeff();
        this->transform_aggregate(retain, exp_op<T>(-max), log_sum_op<T>(+max),
                                  result);
      }
    }

    //! Computes the maximum for each assignment to the given variables.
    void maximum(const domain_type& retain, canonical_array& result) const {
      this->aggregate(retain, max_coeff_op(), result);
    }

    //! Computes the minimum for each assignment to the given variables.
    void minimum(const domain_type& retain, canonical_array& result) const {
      this->aggregate(retain, min_coeff_op(), result);
    }

    //! Returns the normalization constant of the factor.
    logarithmic<T> marginal() const {
      T max = this->param_.maxCoeff();
      // std::log(exp(param()-max).sum())+max is slow (at least on LLVM 3.5)
      T sum = std::accumulate(this->begin(), this->end(), T(0),
                              plus_exp<T>(-max));
      return logarithmic<T>(std::log(sum) + max, log_tag());
    }

    //! Returns the maximum value in the factor.
    logarithmic<T> maximum() const {
      return logarithmic<T>(this->param_.maxCoeff(), log_tag());
    }

    //! Returns the minimum value in the factor.
    logarithmic<T> minimum() const {
      return logarithmic<T>(this->param_.minCoeff(), log_tag());
    }

    //! Computes the maximum value and stores the corresponding assignment.
    logarithmic<T> maximum(finite_assignment& a) const {
      const T* it = std::max_element(this->begin(), this->end());
      this->assignment(it - this->begin(), a);
      return logarithmic<T>(*it, log_tag());
    }

    //! Computes the minimum value and stores the corresponding assignment.
    logarithmic<T> minimum(finite_assignment& a) const {
      const T* it = std::min_element(this->begin(), this->end());
      this->assignment(it - this->begin(), a);
      return logarithmic<T>(*it, log_tag());
    }

    //! Normalizes the factor in-place.
    canonical_array& normalize() {
      this->param_ -= marginal().lv;
      return *this;
    }

    //! Returns true if the factor is normalizable (approximation).
    bool is_normalizable() const {
      return maximum().lv > inf<T>();
    }
    
    //! Restricts this factor to an assignment.
    canonical_array restrict(const finite_assignment& a) const {
      canonical_array result; restrict(a, result);
      return result;
    }

    //! Restricts this factor to an assignment.
    void restrict(const finite_assignment& a, canonical_array& result) const {
      array_factor<T>::restrict(a, result);
    }

    //! Restricts this factor to an assignment and multiplies it into the result.
    void restrict_multiply(const finite_assignment& a,
                           canonical_array& result) const {
      array_factor<T>::restrict_join(a, plus_assign<>(), false, result);
    }

    // Entropy and divergences
    //==========================================================================

    //! Computes the entropy for the distribution represented by this factor.
    T entropy() const {
      return this->transform_accumulate(T(0), entropy_log_op<T>(), std::plus<T>());
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
    friend T cross_entropy(const canonical_array& p,
                           const canonical_array& q) {
      return transform_accumulate(p, q, entropy_log_op<T>(), std::plus<T>());
    }

    //! Computes the Kullback-Liebler divergence from p to q.
    friend T kl_divergence(const canonical_array& p,
                           const canonical_array& q) {
      return transform_accumulate(p, q, kld_log_op<T>(), std::plus<T>());
    }

    //! Computes the Jensen–Shannon divergece between p and q.
    friend T js_divergence(const canonical_array& p,
                           const canonical_array& q) {
      return transform_accumulate(p, q, jsd_log_op<T>(), std::plus<T>());
    }

    //! Computes the sum of absolute differences between the parameters of p and q.
    friend T sum_diff(const canonical_array& p,
                      const canonical_array& q) {
      return transform_accumulate(p, q, abs_difference<T>(), std::plus<T>());
    }
    
    //! Computes the max of absolute differences between the parameters of p and q.
    friend T max_diff(const canonical_array& p,
                      const canonical_array& q) {
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
        g(offset) += w;
      }

      void add_gradient(const param_type& phead, const finite_index& tail, T w,
                        param_type& g) {
        check_shape_compatible(g);
        size_t offset = linear_index(tail, phead.size());
        size_t size = phead.size();
        for (size_t i = 0; i < size; ++i) {
          g(offset + i) += phead(i) * w;
        }
      }
      
      void add_gradient_sqr(const param_type& phead, const finite_index& tail, T w,
                            param_type& g) {
        add_gradient(phead, tail, w, g);
      }

      void add_hessian_diag(const finite_index& index, T w, param_type& h) { }

      void add_hessian_diag(const param_type& phead, const finite_index& tail, T w,
                            param_type& h) { }

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

  }; // class canonical_array

  /**
   * Outputs a human-readable representation of ths factor to the stream.
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const canonical_array<T>& f) {
    out << f.arg_vector() << std::endl
        << f.param() << std::endl;
    return out;
  }

#if 0
    /**
     * Computes the value exp{Exp_q(V)[log f(u, V)]}, where f(U,V) is this
     * factor, and q(V) is the supplied probability_matrix distribution.
     * Multiplies the result to the provided pre-allocated factor h(U).
     */
    void log_exp_mult(const probability_matrix<T>& q, canonical_matrix& h) const {
      assert(q.num_arguments() == 1);
      assert(h.num_arguments() == 1);
      finite_variable* u = h.x_;
      finite_variable* v = q.x_;
      if (x_ == u && y_ == v) {
        h.params_ += params_ * q.values_;
      } else if (x_ == v && y_ == u) {
        h.params_ += params_.t() * q.values_;
      } else {
        throw std::invalid_argument("Unsupported arguments.");
      }
    }
#endif

} // namespace sill

#endif
