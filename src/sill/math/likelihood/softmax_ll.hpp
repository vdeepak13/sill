#ifndef SILL_SOFTMAX_LL_HPP
#define SILL_SOFTMAX_LL_HPP

#include <sill/datastructure/hybrid_index.hpp>
#include <sill/math/param/softmax_param.hpp>
#include <sill/traits/is_sample_range.hpp>

#include <cmath>
#include <type_traits>

#include <Eigen/SparseCore>

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

    /**
     * Creates a log-likelihood function for a probability table with
     * the specified parameters.
     */
    explicit softmax_ll(const softmax_param<T>& f)
      : f(f) { }

    //! Returns the parameters of the log-likelihood function.
    const param_type& param() const {
      return f;
    }
    
    //! Returns the log-likelihood of the label for dense/sparse features.
    template <typename Derived>
    T log(size_t label, const Eigen::EigenBase<Derived>& x) const {
      return std::log(f(x)[label]);
    }

    //! Returns the log-likelihood of the label for sparse unit features.
    T log(size_t label, const std::vector<size_t>& x) const {
      return std::log(f(x)[label]);
    }
     
    //! Returns the log-likelihood of the specified data point.
    T log(const hybrid_index<T>& index) const {
      return std::log(f(index.vector())[index.finite()[0]]);
    }

    /**
     * Adds (expected) gradient of the log-likelihood to g for a datapoint
     * specified as a label and a dense Eigen feature vector.
     * \tparam Label either size_t or a dense Eigen vector of probabilities
     */
    template <typename Label>
    void add_gradient(const Label& label, const dynamic_vector<T>& x, T w,
                      softmax_param<T>& g) const {
      dynamic_vector<T> p = gradient_delta(label, x, w);
      g.weight().noalias() += p * x.transpose();
      g.bias() += p;
    }

    /**
     * Adds (expected) gradient of the log-likelihood to g for a datapoint
     * specified as a label and a sparse Eigen feature vector.
     * \tparam Label either size_t or a dense Eigen vector of probabilities
     */
    template <typename Label>
    void add_gradient(const Label& label, const Eigen::SparseVector<T>& x, T w,
                      softmax_param<T>& g) const {
      dynamic_vector<T> p = gradient_delta(label, x, w);
      for (typename Eigen::SparseVector<T>::InnerIterator it(x); it; ++it) {
        g.weight().col(it.index()) += p * it.value();
      }
      g.bias() += p;
    }

    /**
     * Adds (expected) gradient of the log-likelihood to g for a datapoint
     * specified as a label and a sparse unit feature vector.
     * \tparam Label either size_t or a dense Eigen vector of probabilities
     */
    template <typename Label>
    void add_gradient(const Label& label, const std::vector<size_t>& x, T w,
                      softmax_param<T>& g) const {
      dynamic_vector<T> p = gradient_delta(label, x, w);
      for (size_t i : x) { g.weight().col(i) += p; }
      g.bias() += p;
    }

    /**
     * Adds gradient of the log-likelihood to g for a datapoint
     * specified as hybrid_index.
     */
    void add_gradient(const hybrid_index<T>& x, T w,
                     softmax_param<T>& g) const {
      add_gradient(x.finite()[0], x.vector(), w, g);
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a dense Eigen feature vector.
     */
    void add_hessian_diag(const dynamic_vector<T>& x, T w,
                          softmax_param<T>& h) const {
      dynamic_vector<T> v = hessian_delta(x, w);
      h.weight().noalias() += v * x.cwiseProduct(x).transpose();
      h.bias() += v;
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a sparse Eigen feature vector.
     */
    void add_hessian_diag(const Eigen::SparseVector<T>& x, T w,
                          softmax_param<T>& h) const {
      dynamic_vector<T> v = hessian_delta(x, w);
      for (typename Eigen::SparseVector<T>::InnerIterator it(x); it; ++it) {
        h.weight().col(it.index()) += v * (it.value() * it.value());
      }
      h.bias() += v;
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a sparse feature vector with unit values.
     */
    void add_hessian_diag(const std::vector<size_t>& x, T w,
                          softmax_param<T>& h) const {
      dynamic_vector<T> v = hessian_delta(x, w);
      for (size_t i : x) { h.weight().col(i) += v; }
      h.bias() += v;
    }

    /**
     * Adds the Hessian diagonal of log-likelihood to h for a datapoint
     * specified as a hybrid index.
     */
    void add_hessian_diag(const hybrid_index<T>& x, T w,
                          softmax_param<T>& h) const {
      add_hessian_diag(x.vector(), w, h);
    }

  private:
    template <typename Features>
    dynamic_vector<T>
    gradient_delta(size_t label, const Features& x, T w) const {
      dynamic_vector<T> p = f(x);
      p[label] -= T(1);
      p *= -w;
      return p;
    }

    template <typename Features>
    dynamic_vector<T>
    gradient_delta(const Eigen::Ref<const dynamic_vector<T> >& plabel,
                   const Features& x, T w) const {
      dynamic_vector<T> p = f(x);
      p -= plabel;
      p *= -w;
      return p;
    }

    template <typename Features>
    dynamic_vector<T> hessian_delta(const Features& x, T w) const {
      dynamic_vector<T> v = f(x);
      v -= v.cwiseProduct(v);
      v *= -w;
      return v;
    }

    //! The parameters at which we evaluate the log-likelihood derivatives.
    softmax_param<T> f;

  }; // class softmax_ll

} // namespace sill

#endif
