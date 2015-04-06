#ifndef SILL_SOFTMAX_LL_HPP
#define SILL_SOFTMAX_LL_HPP

#include <sill/datastructure/hybrid_index.hpp>
#include <sill/math/param/softmax_param.hpp>
#include <sill/traits/is_sample_range.hpp>

#include <cmath>
#include <type_traits>

namespace sill {

  /**
   * A log-likelihood function of the softmax distribution and
   * its derivatives.
   *
   * \tparam T the real type representing the coefficients
   */
  template <typename T>
  class softmax_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The table of probabilities.
    typedef softmax_param<T> param_type;

    //! The index type.
    typedef dynamic_vector<T> vec_type;

    /**
     * Creates a log-likelihood function for a probability table with
     * the specified parameters.
     */
    explicit softmax_ll(const softmax_param<T>& f)
      : f(f) { }

    /**
     * Returns the parameters of the log-likelihood function.
     */
    const param_type& param() const {
      return f;
    }
    
    /**
     * Returns the log-likelihood of the specified data point.
     */
    T log(size_t label, const vec_type& x) const {
      return std::log(f(x)[label]);
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T log(const hybrid_index<T>& index) const {
      return std::log(f(index.vector())[index.finite()[0]]);
    }

    /**
     * Returns the log-likelihood of a collection of weighted samples.
     */
    template <typename Range>
    typename std::enable_if<
      is_sample_range<Range, hybrid_index<T>, T>::value, T>::type
    log(const Range& samples) const {
      T result(0);
      for (const auto& sample : samples) {
        result += log(sample.first) * sample.second;
      }
      return result;
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the parameters g.
     */
    void add_gradient(size_t label, const vec_type& x, T w,
                      softmax_param<T>& g) const {
      vec_type p = f(x);
      p[label] -= T(1);
      p *= -w;
      g.weight().noalias() += p * x.transpose();
      g.bias() += p;
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the parameters g.
     */
    void add_gradient(const hybrid_index<T>& x, T w,
                     softmax_param<T>& g) const {
      add_gradient(x.finite()[0], x.vector(), w, g);
    }

    /**
     * Adds the gradient of the expected log-likelihood of the specified
     * data point to the gradient table g.
     *
     * \param plabel the distribution over the labels
     * \param x the featuers
     * \param w the weight of the data point
     */
    void add_gradient(const Eigen::Ref<const vec_type>& plabel,
                      const vec_type& x, T w,
                      softmax_param<T>& g) const {
      vec_type p = f(x);
      p -= plabel;
      p *= -w;
      g.weight().noalias() += p * x.transpose();
      g.bias() += p;
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with features x and weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const vec_type& x, T w, softmax_param<T>& h) const {
      vec_type v = f(x);
      v -= v.cwiseProduct(v);
      v *= -w;
      h.weight().noalias() += v * x.cwiseProduct(x).transpose();
      h.bias() += v;
    }

    /**
     * Adds the diagonal of the Hessian of log-likelihood of the specified
     * data point with features x and weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const hybrid_index<T>& index, T w,
                          softmax_param<T>& h) const {
      add_hessian_diag(index.vector(), w, h);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the parameters g.
     */
    void add_gradient(size_t label, const sparse_index<T>& x, T w,
                      softmax_param<T>& g) const {
      vec_type p = f(x);
      p[label] -= T(1);
      p *= -w;
      for (std::pair<size_t,T> value : x) {
        g.weight().col(value.first) += p * value.second;
      }
      g.bias() += p;
    }

    /**
     * Adds the gradient of the expected log-likelihood of the specified
     * data point to the parameters g.
     *
     * \param plabel the distribution over the labels
     * \param x the featuers
     * \param w the weight of the data point
     */
    void add_gradient(const Eigen::Ref<const vec_type>& plabel,
                      const sparse_index<T>& x, T w,
                      softmax_param<T>& g) const {
      vec_type p = f(x);
      p -= plabel;
      p *= -w;
      for (std::pair<size_t,T> value : x) {
        g.weight().col(value.first) += p * value.second;
      }
      g.bias() += p;
    }

    /**
     * Adds the diagonal of the Hessian of log-likleihood of the specified
     * data point with features x and weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(const sparse_index<T>& x, T w,
                          softmax_param<T>& h) const {
      vec_type v = f(x);
      v -= v.cwiseProduct(v);
      v *= -w;
      for (std::pair<size_t,T> value : x) {
        h.weight().col(value.first) += v * (value.second * value.second);
      }
      h.bias() += v;
    }

  private:
    //! The parameters at which we evaluate the log-likelihood derivatives.
    softmax_param<T> f;

  }; // class softmax_ll

} // namespace sill

#endif
