#ifndef SILL_SOFTMAX_HPP
#define SILL_SOFTMAX_HPP

#include <sill/datastructure/hybrid_index.hpp>
#include <sill/datastructure/sparse_index.hpp>

#include <armadillo>

#include <cmath>
#include <iostream>

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
  class softmax {
  public:
    // Public types
    //======================================================================
    typedef arma::Mat<T> mat_type;
    typedef arma::Col<T> vec_type;
    typedef hybrid_index<T> index_type;

    // Constructors
    //======================================================================

    /**
     * Creates an empty softmax. This does not represent a valid function.
     */
    softmax() { }

    /**
     * Creates a softmax function with the given number of labels and
     * features. Allocates the parameters, but does not initialize them
     * to any specific value.
     */
    softmax(size_t num_labels, size_t num_features)
      : bias_(num_labels), weight_(num_labels, num_features) { }

    /**
     * Creates a softmax function with the given number weight matrix
     * and bias vector.
     */
    softmax(const mat_type& weight, const vec_type& bias)
      : weight_(weight), bias_(bias) {
      assert(weight.n_rows == bias.n_rows);
    }

    /**
     * Swaps the content of two factors.
     */
    friend void swap(const softmax& f, const softmax& g) {
      f.weight_.swap(g.weight_);
      f.bias_.swap(g.bias_);
    }

    /**
     * Resets the function to the given number of labels and features.
     * May invalidate the parameters.
     */
    void reset(size_t num_labels, size_t num_features) {
      weight_.set_size(num_labels, num_features);
      bias_.set_size(num_labels);
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns true if the function is empty.
    bool empty() const {
      return bias_.empty();
    }

    //! Returns the number of labels.
    size_t num_labels() const {
      return weight_.n_rows;
    }

    //! Returns the numebr of features.
    size_t num_features() const {
      return weight_.n_cols;
    }

    //! Returns the weight matrix.
    const mat_type& weight() const {
      weight_;
    }

    //! Returns the bias vector.
    const vec_type& bias() const {
      bias_;
    }

    //! Evaluates the function for a dense feature vector.
    vec_type operator()(const vec_type& x) const {
      vec_type y = exp(weight_ * x + bias_);
      return y /= sum(y);
    }

    //! Evaluates the function for a sparse feature vector.
    vec_type operator()(const sparse_index<T>& x) const {
      vec_type y = bias_;
      for (std::pair<size_t,T> value : x) {
        y += weight_.col(value.first) * value.second;
      }
      y = exp(y);
      return y /= sum(y);
    }

    // OptimizationVector functions
    //=========================================================================
    softmax& operator+=(const softmax& f) {
      weight_ += f.weight_;
      bias_ += f.bias_;
      return *this;
    }

    softmax& operator-=(const softma& f) {
      weight_ -= f.weight_;
      bias_ -= f.bias_;
      return *this;
    }

    softmax& operator*=(T a) {
      weight_ *= a;
      bias_ *= a;
      return *this;
    }

    softmax& operator/=(T a) {
      weight_ /= a;
      bias_ /= a;
      return *this;
    }

    friend void axpy(T a, const softmax& x, softmax& y) {
      y.weight_ += a * x.weight_;
      y.bias_ += a * x.bias_;
    }

    friend T dot(const softmax& f, const softmax& g) {
      return dot(f.weight_, g.weight_) + dot(f.bias_, g.bias_);
    }

    // LogLikelihoodDerivatives functions
    //=========================================================================
    void add_gradient(const softmax& f,
                      size_t label, const vec_type& x, T w) {
      vec_type p = f(x);
      p[label] -= T(1);
      p *= w;
      weight_ += p * x.t();
      bias_ += p;
    }

    void add_gradient(const softmax& f,
                      const vec_type& plabel, const vec_type& x, T w) {
      vec_type p = f(x);
      p -= plabel;
      p *= w;
      weight_ += p * x.t();
      bias_ += p;
    }

    void add_hessian_diag(const softmax& f, const vec_type& x, T w) {
      vec_type v = f(x);
      v -= v % v;
      v *= w;
      weight_ += p * trans(x % x);
      bias_ += v;
    }

    void add_gradient(const softmax& f,
                      size_t label, const sparse_index<T>& x, T w) {
      vec_type p = f(x);
      p[label] -= T(1);
      p *= w;
      for (std::pair<size_t,T> value : x) {
        weight_.col(value.first) += p * value.second;
      }
      bias_ += p;
    }

    void add_gradient(const softmax& f,
                      const vec_type& plabel, const sparse_index<T>& x, T w) {
      vec_type p = f(x);
      p -= plabel;
      p *= w;
      for (std::pair<size_t,T> value : x) {
        weight_.col(value.first) += p * value.second;
      }
      bias_ += p;
    }

    void add_hessian_diag(const softmax& f, const sparse_index<T>& x, T w) {
      vec_type v = f(x);
      v -= v % v;
      v *= w;
      for (std::pair<size_t,T> value : x) {
        weight_.col(value.first) += v * (value.second * value.second);
      }
      bias_ += v;
    }

  private:
    // Private members
    //=========================================================================
    mat_type weight_;
    vec_type bias_;
    
  }; // class soft_max
  
  /**
   * Prints the softmax function parameters to a stream.
   * \relates softmax
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const softmax<F>& f) {
    out << join_horiz(f.weight(), f.bias());
    return out;
  }

} // namespace sill

#endif
