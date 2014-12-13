#ifndef SILL_CANONICAL_MATRIX_HPP
#define SILL_CANONICAL_MATRIX_HPP

#include <sill/global.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/factor/probability_matrix.hpp>
#include <sill/math/logarithmic.hpp>

#include <armadillo>

#include <boost/function.hpp>

#include <iostream>

namespace sill {

  /**
   * A table-like factor of a categorical probability distribution
   * represented in the canonical form of the exponential family.
   * This factor represents a non-negative function over finite
   * variables X as f(X | \theta) = exp(\sum_x \theta_x * 1(X=x)).
   * In some cases, as with factors comprising Bayesian networks, or
   * beliefs in inference algorithms, this factor also represents
   * a probability distribution in the log-space.
   *
   * This class is implemented using Armadillo's matrix type, and
   * currently only supports a limited set of operations needed
   * for mean Field.
   *
   * \tparam T type that represents the natural parameters
   */
  template <typename T = double>
  class canonical_matrix {
    // Public types
    //==========================================================================
  public:
    // Factor member types
    typedef T                 real_type;
    typedef logarithmic<T>    result_type;
    typedef finite_variable   variable_type;
    typedef finite_domain     domain_type;
    typedef finite_var_vector var_vector_type;
    typedef finite_assignment assignment_type;
    
    // IndexableFactor member types
    typedef std::vector<size_t> index_type;
    
    // DistributionFactor member types
    typedef boost::function<canonical_matrix(const finite_domain&)>
      marginal_fn_type;
    typedef boost::function<canonical_matrix(const finite_domain&,
                                             const finite_domain&)>
      conditional_fn_type;
    
    // LearnableFactor types
    typedef finite_dataset dataset_type;
    typedef finite_record record_type;

    // ExponentialFactor types
    typedef probability_matrix<T> probability_factor_type;

    // Underlying representation
    typedef arma::Mat<T> mat_type;

    // Constructors and conversion operators
    //==========================================================================
  public:
    /**
     * Constructs a canonical_matrix factor with no arguments that
     * represents a constant value.
     */
    explicit canonical_matrix(logarithmic<T> value = 1.0)
      : x_(NULL), y_(NULL) {
      params_ = log(value);
    }
    
    /**
     * Constructs a canonical_matrix factor with the given arguments
     * that represents a constant value.
     */
    explicit canonical_matrix(const finite_domain& args,
                              logarithmic<T> value = 1.0) {
      initialize(make_vector(args));
      params_.fill(log(value));
    }

    /**
     * Constructs a canonical_matrix factor with the given arguments
     * that represents a constant value.
     */
    explicit canonical_matrix(const finite_var_vector& args,
                              logarithmic<T> value = 1.0) {
      initialize(args);
      params_.fill(log(value));
    }

    /**
     * Constructs a canonical_matrix factor with a single argument.
     * Allocates the memory, but does _not_ initialize the parameters.
     */
    explicit canonical_matrix(finite_variable* x)
      : x_(x), y_(NULL), params_(x->size(), 1) {
      args_.insert(x);
    }

    /**
     * Constructs a canonical_matrix factor with two arguments.
     * Allocates the memory, but does _not_ initialize the parameters.
     */
    canonical_matrix(finite_variable* x, finite_variable* y)
      : x_(x), y_(y), params_(x->size(), y->size()) {
      args_.insert(x);
      args_.insert(y);
    }

    /**
     * Constructs a canonical_matrix factor equivalent to the given
     * probability_matrix factor.
     */
    explicit canonical_matrix(const probability_matrix<T>& f) 
      : args_(f.args_), x_(f.x_), y_(f.y_) {
      params_ = log(f.values_);
    }

    /**
     * Constructs a canonical_matrix factor equivalent to the given
     * table factor.
     */
    explicit canonical_matrix(const table_factor& f) {
      initialize(f.arg_vector());
      assert(size() == f.size());
      T (*fn)(T) = &std::log;
      std::transform(f.begin(), f.end(), params_.begin(), fn);
    }      

    /**
     * Assigns a constant to this factor.
     */
    canonical_matrix& operator=(logarithmic<T> value) {
      args_.clear();
      x_ = NULL;
      y_ = NULL;
      params_ = log(value);
      return *this;
    }

    /**
     * Assigns a probability_matrix factor to this factor.
     */
    canonical_matrix& operator=(const probability_matrix<T>& f) {
      args_ = f.args_;
      x_ = f.x_;
      y_ = f.y_;
      params_ = log(f.values_);
      return *this;
    }

    /**
     * Assigns a table_factor to this factor.
     */
    canonical_matrix& operator=(const table_factor& f) {
      initialize(f.arg_vector());
      assert(size() == f.size());
      T (*fn)(T) = &std::log;
      std::transform(f.begin(), f.end(), params_.begin(), fn);
      return *this;
    }

    /**
     * Swaps the content of two canonical_matrix factors.
     */
    void swap(canonical_matrix& other) {
      args_.swap(other.args_);
      std::swap(x_, other.x_);
      std::swap(y_, other.y_);
      params_.swap(other.params_);
    }

    // Accessors
    //==========================================================================
    //! Returns the argument set of this factor
    const finite_domain& arguments() const {
      return args_;
    }

    //! Returns the argument vector of this factor
    finite_var_vector arg_vector() const {
      return make_vector(x_, y_);
    }

    //! Returns the number of arguments of this factor
    size_t num_arguments() const {
      return args_.size();
    }

    //! Returns the parameter matrix.
    const mat_type& params() const {
      return params_;
    }
    
    //! Returns the parameter matrix. The caller must not alter the dimensions.
    mat_type& params() {
      return params_;
    }

    //! Returns the number of elements of the factor
    size_t size() const {
      return params_.size();
    }

    //! Returns the pointer to the first element
    const T* begin() const {
      return params_.begin();
    }

    //! Returns the pointer to the one past the last element
    const T* end() const {
      return params_.end();
    }

    //! Returns the i-th element in the linear indexing
    const T& operator[](size_t i) const {
      return params_(i);
    }

    //! Returns the i-th element in the linear indexing
    T& operator[](size_t i) {
      return params_(i);
    }

    //! Returns the value of the factor for the given finite assignment
    logarithmic<T> operator()(const finite_assignment& a) const {
      return logarithmic<T>(logv(a), log_tag());
    }

    //! Returns the value of the factor for the given index
    logarithmic<T> operator()(const index_type& index) const {
      return logarithmic<T>(logv(index), log_tag());
    }

    //! Returns the log-value of the factor for the given assignment
    T logv(const finite_assignment& a) const {
      return params_(x_ ? safe_get(a, x_) : 0, y_ ? safe_get(a, y_) : 0);
    }

    //! Returns the log-value of the factor for the given index
    T logv(const index_type& index) const {
      return params_(x_ ? index[0] : 0, y_ ? index[1] : 0);
    }

    // Factor operations
    //==========================================================================
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

    //! Returns the normalization constant of this factor.
    logarithmic<T> norm_constant() const {
      return logarithmic<T>(accu(exp(params_)));
    }

    //! Ensures that the factor sums to 1.
    canonical_matrix& normalize() {
      params_ -= log(norm_constant());
      return *this;
    }

    // Private functions and data members
    //==========================================================================
  private:
    /**
     * Initializes the finite arguments to the given vector.
     * Resizes the prameter matrix, but does not initialize its elements.
     */
    void initialize(const finite_var_vector& args) {
      args_.clear();
      args_.insert(args.begin(), args.end());
      switch (args.size()) {
      case 0:
        x_ = NULL;
        y_ = NULL;
        params_.set_size(1,1);
        return;
      case 1:
        x_ = args[0];
        y_ = NULL;
        params_.set_size(x_->size(), 1);
        return;
      case 2:
        x_ = args[0];
        y_ = args[1];
        params_.set_size(x_->size(), y_->size());
        return;
      default:
        throw std::invalid_argument(
          "canonical_matrix only supports <= arguments"
        );
      }
    }

    //! The argument set of this factor
    finite_domain args_;

    //! The row argument of this factor (may be NULL if y_ is also NULL)
    finite_variable* x_;

    //! The column argument of this factor (may be NULL)
    finite_variable* y_;

    //! The natural parameters representing this factor
    mat_type params_;

    friend class probability_matrix<T>;

  }; // class canonical_matrix

  /**
   * Outputs a human-readable representation of the factor to stream.
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const canonical_matrix<T>& f) {
    out << f.arg_vector() << std::endl
        << (f.num_arguments() == 1) ? f.params().t() : f.params() << std::endl;
    return out;
  }

} // namespace sill

#endif
