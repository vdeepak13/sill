#ifndef SILL_SOFTMAX_PARAM_HPP
#define SILL_SOFTMAX_PARAM_HPP

#include <sill/datastructure/hybrid_index.hpp>
#include <sill/datastructure/sparse_index.hpp>
#include <sill/math/eigen/dynamic.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace sill {

  /**
   * A softmax function over one discrete variable y and a vector of
   * real-valued features x. This function is equal to a normalized
   * exponential, f(y=i, x) = exp(b_i + w_i^T x) / sum_j exp(b_j + w_j^T x).
   * Here, b is a bias vector and w is a weight matrix with rows w_i^T.
   * The parameter matrices are dense, but the function can be evaluated
   * on sparse feature vectors.
   *
   * This class models the OptimizationVector concept and can be directly
   * used in optimization classes.
   *
   * \tparam T a real type for representing each parameter.
   * \ingroup math_functions
   */
  template <typename T>
  class softmax_param {
  public:
    // Public types
    //======================================================================
    // OptimizationVector types
    typedef T value_type;

    // Underlying representation
    typedef dynamic_matrix<T> mat_type;
    typedef dynamic_vector<T> vec_type;

    // Constructors
    //======================================================================

    /**
     * Creates an empty softmax. This does not represent a valid function.
     */
    softmax_param() { }

    /**
     * Creates a softmax function with the given number of labels and
     * features. Allocates the parameters, but does not initialize them
     * to any specific value.
     */
    softmax_param(size_t num_labels, size_t num_features)
      : weight_(num_labels, num_features), bias_(num_labels) { }

    /**
     * Creates a softmax function with the given number of labels and
     * features, and initializes the parameters to the given value.
     */
    softmax_param(size_t num_labels, size_t num_features, T init)
      : weight_(num_labels, num_features), bias_(num_labels) {
      bias_.fill(init);
      weight_.fill(init);
    }

    /**
     * Creates a softmax function with the given parameters.
     */
    softmax_param(const mat_type& weight, const vec_type& bias)
      : weight_(weight), bias_(bias) {
      assert(weight.rows() == bias.rows());
    }

    /**
     * Creates a softmax function with the given parameters.
     */
    softmax_param(mat_type&& weight, vec_type&& bias) {
      weight_.swap(weight);
      bias_.swap(bias);
      assert(weight.rows() == bias.rows());
    }

    //! Copy constructor.
    softmax_param(const softmax_param& other) = default;

    //! Move constructor.
    softmax_param(softmax_param&& other) {
      swap(*this, other);
    }

    //! Assignment operator.
    softmax_param& operator=(const softmax_param& other) {
      if (this != &other) {
        weight_ = other.weight_;
        bias_ = other.bias_;
      }
      return *this;
    }

    //! Move assignment operator.
    softmax_param& operator=(softmax_param&& other) {
      swap(*this, other);
      return *this;
    }
    
    //! Swaps the content of two softmax functions.
    friend void swap(softmax_param& f, softmax_param& g) {
      f.weight_.swap(g.weight_);
      f.bias_.swap(g.bias_);
    }

    /**
     * Resets the function to the given number of labels and features.
     * May invalidate the parameters.
     */
    void resize(size_t num_labels, size_t num_features) {
      weight_.resize(num_labels, num_features);
      bias_.resize(num_labels);
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns true if the softmax function is empty.
    bool empty() const {
      return !weight_.data();
    }

    //! Returns the number of labels.
    size_t num_labels() const {
      return weight_.rows();
    }

    //! Returns the number of features.
    size_t num_features() const {
      return weight_.cols();
    }

    //! Returns the weight matrix.
    mat_type& weight() {
      return weight_;
    }

    //! Returns the weight matrix.
    const mat_type& weight() const {
      return weight_;
    }

    //! Returns the bias vector.
    vec_type& bias() {
      return bias_;
    }

    //! Returns the bias vector.
    const vec_type& bias() const {
      return bias_;
    }

    //! Returns the weight with the given indices.
    T& weight(size_t i, size_t j) {
      return weight_(i, j);
    }

    //! Returns the weight with the given indices.
    const T& weight(size_t i, size_t j) const {
      return weight_(i, j);
    }

    //! Returns the bias with the given index.
    T& bias(size_t i) {
      return bias_[i];
    }

    //! Returns the bias with the given index.
    const T& bias(size_t i) const {
      return bias_[i];
    }

    //! Evaluates the function for a dense feature vector.
    vec_type operator()(const vec_type& x) const {
      assert(x.size() == weight_.cols());
      vec_type y(exp((weight_ * x + bias_).array()).matrix());
      y /= y.sum();
      return y;
    }

    //! Evaluates the function for a sparse feature vector.
    vec_type operator()(const sparse_index<T>& x) const {
      vec_type y = bias_;
      for (std::pair<size_t,T> value : x) {
        assert(value.first < weight_.cols());
        y += weight_.col(value.first) * value.second;
      }
      y = exp(y.array()).matrix();
      y /= y.sum();
      return y;
    }

    //! Returns the log-value for a dense feature vector.
    vec_type log(const vec_type& x) const {
      vec_type y = operator()(x);
      y = y.array().log();
      return y;
    }

    //! Returns the log-value for a sparse feature vector.
    vec_type log(const sparse_index<T>& x) const {
      vec_type y = operator()(x);
      y = y.array().log();
      return y;
    }

    //! Returns true if all the parameters are finite and not NaN.
    bool is_finite() const {
      return weight_.allFinite() && bias_.allFinite();
    }

    //! Returns true if two softmax parameter vectors are equal.
    friend bool operator==(const softmax_param& f, const softmax_param& g) {
      return f.weight_ == g.weight_ && f.bias_ == g.bias_;
    }

    //! Returns true if two softmax parameter vecors are not equal.
    friend bool operator!=(const softmax_param& f, const softmax_param& g) {
      return !(f == g);
    }

    // Sampling
    //==========================================================================
    template <typename Generator>
    size_t sample(Generator& rng, const vec_type& x) const {
      vec_type p = operator()(x);
      T val = std::uniform_real_distribution<T>()(rng);
      for (size_t i = 0; i < p.size(); ++i) {
        if (val <= p[i]) {
          return i;
        } else {
          val -= p[i];
        }
      }
      throw std::logic_error("The probabilities do not sum to 1");
    }

    // OptimizationVector functions
    //=========================================================================
    void zero() {
      weight_.fill(0);
      bias_.fill(0);
    }

    softmax_param operator-() const {
      return softmax_param(-weight_, -bias_);
    }

    softmax_param& operator+=(const softmax_param& f) {
      weight_ += f.weight_;
      bias_ += f.bias_;
      return *this;
    }

    softmax_param& operator-=(const softmax_param& f) {
      weight_ -= f.weight_;
      bias_ -= f.bias_;
      return *this;
    }

    softmax_param& operator/=(const softmax_param& f) {
      weight_.array() /= f.weight_.array();
      bias_.array() /= f.bias_.array();
      return *this;
    }

    softmax_param& operator+=(T a) {
      weight_.array() += a;
      bias_.array() += a;
      return *this;
    }

    softmax_param& operator-=(T a) {
      weight_.array() -= a;
      bias_.array() -= a;
      return *this;
    }

    softmax_param& operator*=(T a) {
      weight_ *= a;
      bias_ *= a;
      return *this;
    }

    softmax_param& operator/=(T a) {
      weight_ /= a;
      bias_ /= a;
      return *this;
    }

    friend void axpy(T a, const softmax_param& x, softmax_param& y) {
      y.weight_ += a * x.weight_;
      y.bias_ += a * x.bias_;
    }

    friend T dot(const softmax_param& f, const softmax_param& g) {
      return f.weight_.cwiseProduct(g.weight_).sum() + f.bias_.dot(g.bias_);
    }

  private:
    // Private members
    //=========================================================================

    //! The weight matrix.
    mat_type weight_;

    //! The bias vector.
    vec_type bias_;
    
  }; // class softmax_param
  
  /**
   * Prints the softmax function parameters to a stream.
   * \relates softmax_param
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const softmax_param<T>& f) {
    typename softmax_param<T>::mat_type a(f.num_labels(), f.num_features() + 1);
    a << f.weight(), f.bias();
    out << a << std::endl;
    return out;
  }

} // namespace sill

#endif
