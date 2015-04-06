#ifndef SILL_CANONICAL_ARRAY_LL_HPP
#define SILL_CANONICAL_ARRAY_LL_HPP

#include <sill/global.hpp>
#include <sill/datastructure/finite_index.hpp>
#include <sill/traits/is_sample_range.hpp>

#include <Eigen/Core>

namespace sill {

  /**
   * A log-likelihood function of an array distribution in the natural
   * (canonical) parameterization and its derivatives.
   *
   * \tparam T the real type representing the parameters
   * \tparam N the arity of the distribution (1 or 2)
   */
  template <typename T, size_t N>
  class canonical_array_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The array of natural parameters.
    typedef Eigen::Array<T, Eigen::Dynamic, N == 1 ? 1 : Eigen::Dynamic> param_type;

    //! The 1-D array of probabilities.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> array1_type;

    /**
     * Constructs a log-likelihood function for a canonical array
     * with the specified parameters.
     */
    explicit canonical_array_ll(const param_type& a)
      : a(a) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T log(size_t i) const {
      return a(i);
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T log(size_t i, size_t j) const {
      return a(i, j);
    }

    /**
     * Returns the log-likelihood of a collection of weighted samples.
     */
    template <typename Range>
    typename std::enable_if<
      is_sample_range<Range, finite_index, T>::value, T>::type
    log(const Range& samples) const {
      T result(0);
      for (const auto& sample : samples) {
        result += a(linear(sample.first)) * sample.second;
      }
      return result;
    }

    /**
     * Returns the log-likelihood of a collection of weighted samples.
     */
    template <typename Range>
    typename std::enable_if<
      is_sample_range<Range, size_t, T>::value && (N == 1), T>::type
    log(const Range& samples) const {
      T result(0);
      for (const auto& sample : samples) {
        result += a[sample.first] * sample.second;
      }
      return result;
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g.
     * \param i the row index
     */
    void add_gradient(size_t i, T w, param_type& g) const {
      g(i) += w;
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g
     * \param i the row index
     * \param j the column index
     */
    void add_gradient(size_t i, size_t j, T w, param_type& g) const {
      g(i, j) += w;
    }

    /**
     * Adds a gradient of the expected log-likelihood of the specified
     * data point to the gradient table g.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_gradient(const array1_type& phead, size_t j, T w,
                      param_type& g) const {
      g.col(j) += w * phead;
    }
      
    /**
     * Adds the diagonal of the Hessian of log-likleihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(size_t i, T w, param_type& h) const { }

    /**
     * Adds the diagonal of the Hessian of log-likleihood of the specified
     * data point with weight w to the Hessian diagonal h.
     */
    void add_hessian_diag(size_t i, size_t j, T w, param_type& h) const { }

    /**
     * Adds the diagonal of the Hessian of the expected log-likelihoood of
     * the specified data point to the Hessian diagonal h.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_hessian_diag(const array1_type& phead, size_t j, T w,
                          param_type& h) const { }

  private:
    //! Returns the linear index corresponding to a finite_index.
    size_t linear(const finite_index& index) const {
      switch (index.size()) {
      case 1:
        assert(a.cols() == 1);
        return index[0];
      case 2:
        assert(index[1] < a.cols());
        return index[0] + index[1] * a.rows();
      default:
        throw std::invalid_argument("Invalid length of the finite index");
      }
    }

    //! The parameters at which we evaluate the log-likelihood derivatives.
    param_type a;
    
  }; // class canonical_array_ll

} //namespace sill

#endif
