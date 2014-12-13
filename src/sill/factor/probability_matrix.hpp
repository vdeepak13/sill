#ifndef SILL_PROBABILITY_MATRIX_HPP
#define SILL_PROBABILITY_MATRIX_HPP

#include <sill/global.hpp>
#include <sill/functional.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/finite_dataset.hpp>

#include <armadillo>
#include <boost/function.hpp>

#include <iostream>

namespace sill {

  // Forward declaration
  template <typename T> class canonical_matrix;

  /**
   * A table-like factor of a categorical probability distribution that may
   * contain up to 2 arguments in its domain. In some cases, this class
   * represents a matrix of probabilities (e.g., when used in a Bayesian
   * network). In others, this class merely represents a potential that
   * contributes to the factorized form of the distribution.
   * 
   * The class is implemented using Armadillo's matrix type. It supports
   * factor operations very efficiently.
   *
   * \tparam T a type of values stored in the factor
   */
  template <typename T = double>
  class probability_matrix {
  public:
    // Factor member types
    typedef T                 result_type;
    typedef T                 real_type;
    typedef finite_variable   variable_type;
    typedef finite_domain     domain_type;
    typedef finite_var_vector var_vector_type;
    typedef finite_assignment assignment_type;
    
    // IndexableFactor member types
    typedef std::vector<size_t> index_type;
    
    // DistributionFactor member types
    typedef boost::function<probability_matrix(const finite_domain&)>
      marginal_fn_type;
    typedef boost::function<probability_matrix(const finite_domain&,
                                               const finite_domain&)>
      conditional_fn_type;
    
    // LearnableFactor types
    typedef finite_dataset dataset_type;

    // Underlying representation
    typedef arma::Mat<T> mat_type;
    typedef const T* const_iterator;
    typedef T* iterator;

    // Constructors and conversion operators
    //==========================================================================
  public:
    /**
     * Constructs a probability_matrix factor with no arguments that
     * represents a constant value.
     */
    explicit probability_matrix(T value = 1.0)
      : x_(NULL), y_(NULL) {
      values_ = value;
    }

    /**
     * Constructs a probability_matrix factor with given argument set that
     * represents a constant value.
     */
    explicit probability_matrix(const finite_domain& args, T value = 1.0) {
      initialize(make_vector(args));
      values_.fill(value);
    }
  
    /**
     * Constructs a probability_matrix factor with given argument vector that
     * represents a constant value.
     */ 
    explicit probability_matrix(const finite_var_vector& args, T value = 1.0) {
      initialize(args);
      values_.fill(value);
    }

    /**
     * Constructs a probability_matrix factor with a single argument.
     * Allocates the memory, but does not initialize the values.
     */
    explicit probability_matrix(finite_variable* x)
      : x_(x), y_(NULL), values_(x->size(), 1) {
      args_.insert(x);
    }

    /**
     * Constructs a probability_matrix factor with two arguments.
     * Allocates teh memory, but does not initialize the values.
     */
    probability_matrix(finite_variable* x, finite_variable* y)
      : x_(x), y_(y), values_(x->size(), y->size()) {
      args_.insert(x);
      args_.insert(y);
    }

    /**
     * Constructs a probability_matrix factor equivalent to the given
     * canonical_matrix factor.
     */
    explicit probability_matrix(const canonical_matrix<T>& f)
      : args_(f.args_), x_(f.x_), y_(f.y_) {
      values_ = exp(f.params_);
    }

    /**
     * Constructs a probability matrix factor equivalent to the given
     * table factor.
     */
    explicit probability_matrix(const table_factor& f) {
      initialize(f.arg_vector());
      assert(size() == f.size());
      std::copy(f.begin(), f.end(), values_.begin());
    }

    /**
     * Assigns a constant to this factor.
     */
    probability_matrix& operator=(T value) {
      x_ = NULL;
      y_ = NULL;
      args_.clear();
      values_ = value;
      return *this;
    }

    /**
     * Assigns a canonical_matrix factor to this factor.
     */
    probability_matrix& operator=(const canonical_matrix<T>& f) {
      args_ = f.args_;
      x_ = f.x_;
      y_ = f.y_;
      values_ = exp(f.params_);
      return *this;
    }
    
    /**
     * Assigns a table factor to this factor.
     */
    probability_matrix& operator=(const table_factor& f) {
      initialize(f.arg_vector());
      assert(size() == f.size());
      std::copy(f.begin(), f.end(), values_.begin());
      return *this;
    }

    /**
     * Swaps the content of two probability_matrix factors.
     */
    void swap(probability_matrix& f) {
      args_.swap(f.args_);
      std::swap(x_, f.x_);
      std::swap(y_, f.y_);
      values_.swap(f.values_);
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

    //! Returns the number of arguments of this factor.
    size_t num_arguments() const {
      return args_.size();
    }

    //! Returns the values of this factor
    const mat_type& values() const {
      return values_;
    }

    //! Returns the values matrix. The caller must not modify its dimensions.
    mat_type& values() {
      return values_;
    }

    //! Returns the number of elements of the factor.
    size_t size() const {
      return values_.size();
    }

    //! Returns the pointer to the first element
    const T* begin() const {
      return values_.begin();
    }

    //! Returns the pointer to the first element
    T* begin() {
      return values_.begin();
    }

    //! Returns the pointer to the one past the last element
    const T* end() const {
      return values_.end();
    }

    //! Returns the pointer ot the one past the last element
    T* end() {
      return values_.end();
    }

    //! Returns the i-th element in the linear indexing
    const T& operator[](size_t i) const {
      return values_(i);
    }

    //! Returns the i-th element in the linear indexing
    T& operator[](size_t i) {
      return values_(i);
    }

    //! Returns the value of this factor for an assignment
    T operator()(const finite_assignment& a) const {
      return values_(x_ ? safe_get(a, x_) : 0, y_ ? safe_get(a, y_) : 0);
    }

    //! Returns the value of this factor for an index
    T operator()(const index_type& index) const {
      return values_(x_ ? index[0] : 0, y_ ? index[1] : 0);
    }

    //! Returns the logarithm of the factor value for an assignment
    T logv(const finite_assignment& a) const {
      return log(operator()(a));
    }

    //! Returns the logairthm of the factor value for an index
    T logv(const index_type& index) const {
      return log(operator()(index));
    }

    // Factor operations
    //==========================================================================
    //! Returns the normalization constant of this factor
    T norm_constant() const {
      return accu(values_);
    }
    
    //! Ensures that the factor values sum to 1.
    probability_matrix& normalize() {
      values_ /= norm_constant();
      return *this;
    }

    /**
     * Transforms two factors and sums up the result. The two factors must
     * have the same argument sets.
     */
    template <typename Op>
    static T join_sum(const probability_matrix& f,
                      const probability_matrix& g,
                      Op op) {
      assert(f.arguments() == g.arguments());
      T result = 0.0;
      const T* f_it = f.begin();
      if (f.x_ == g.x_) { // y_ will match, too
        const T* g_it = g.begin();
        for (; f_it != f.end(); ++f_it, ++g_it) {
          result += op(*f_it, *g_it);
        }
      } else {            // x_ and y_ are swapped in f vs. g
        mat_type gt = g.values_.t();
        const T* g_it = gt.begin();
        for (; f_it != f.end(); ++f_it, ++g_it) {
          result += op(*f_it, *g_it);
        }
      }
      return result;
    }
   
    /**
     * Computes the relative entropy from this factor (p) to the given 
     * factor (q). Both factors must represent a normalized probability
     * distribution.
     */
    real_type relative_entropy(const probability_matrix& q) const {
      return join_sum(*this, q, kld_operator<T>());
    }
    
    /**
     * Computes the cross etrnopy from this factor (p) to the given
     * factor (q). Both factors must represent a normalized probability
     * distribution.
     */
    real_type cross_entropy(const probability_matrix& q) const {
      return join_sum(*this, q, cross_entropy_operator<T>());
    }

    // Private functions and data members
    //==========================================================================
  private:
    /**
     * Initializes the finite arguments to the given vector.
     * Resizes the values matrix, but does not initialize its elements.
     */
    void initialize(const finite_var_vector& args) {
      args_.clear();
      args_.insert(args.begin(), args.end());
      switch (args.size()) {
      case 0:
        x_ = NULL;
        y_ = NULL;
        values_.set_size(1, 1);
        return;
      case 1:
        x_ = args[0];
        y_ = NULL;
        values_.set_size(x_->size(), 1);
        return;
      case 2:
        x_ = args[0];
        y_ = args[1];
        values_.set_size(x_->size(), y_->size());
        return;
      default:
        throw std::invalid_argument("probability_matrix supports only <= 2 arguments");
      }
    }

    //! The argument set of this factor
    finite_domain args_;

    //! The row argument of this factor (may be NULL if y_ is also NULL)
    finite_variable* x_;
    
    //! The column argument of this factor
    finite_variable* y_;

    //! The values of this factor
    mat_type values_;

    friend class canonical_matrix<T>;

  }; // class probability_matrix

  /**
   * Outputs a human-readable representation of ths factor to the stream.
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const probability_matrix<T>& f) {
    out << f.arg_vector() << std::endl
        << (f.num_arguments() == 1) ? f.values().t() : f.values() << std::endl;
    return out;
  }

  /**
   * Returns the sum of absolute differences between the probabilities.
   */
  template <typename T>
  T diff_1(const probability_matrix<T>& f, const probability_matrix<T>& g) {
    return probability_matrix<T>::join_sum(f, g, abs_difference<T>());
  }

} // namespace sill

#endif
