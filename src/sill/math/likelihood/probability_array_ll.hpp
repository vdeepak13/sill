#ifndef SILL_PROBABILITY_ARRAY_LL_HPP
#define SILL_PROBABILITY_ARRAY_LL_HPP

#include <sill/global.hpp>
#include <sill/datastructure/finite_index.hpp>
#include <sill/traits/is_sample_range.hpp>

#include <stdexcept>

#include <Eigen/Core>

namespace sill {

  /**
   * A log-likelihood function of an probability array and its derivatives.
   *
   * \tparam T the real type representing the parameters
   * \tparam N the arity of the distribution (1 or 2)
   */
  template <typename T, size_t N>
  class probability_array_ll {
  public:
    //! The real type representing the log-likelihood.
    typedef T real_type;

    //! The regularization parameter type.
    typedef T regul_type;

    //! The array of probabilities.
    typedef Eigen::Array<T, Eigen::Dynamic, N == 1 ? 1 : Eigen::Dynamic> param_type;

    //! The 1-D array of probabilities.
    typedef Eigen::Array<T, Eigen::Dynamic, 1> array1_type;

    /**
     * Constructs a log-likelihood function for a probability array
     * with the specified parameters (probabilities).
     */
    explicit probability_array_ll(const param_type& a)
      : a(a) { }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T log(size_t i) const {
      return std::log(a(i));
    }

    /**
     * Returns the log-likelihood of the specified data point.
     */
    T log(size_t i, size_t j) const {
      return std::log(a(i, j));
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
        result += std::log(a(linear(sample.first))) * sample.second;
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
        result += std::log(a[sample.first]) * sample.second;
      }
      return result;
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g.
     * \param i the row index
     */
    void add_gradient(size_t i, T w, param_type& g) const {
      g(i) += w / a(i);
    }

    /**
     * Adds a gradient of the log-likelihood of the specified data
     * point with weight w to the gradient array g
     * \param i the row index
     * \param j the column index
     */
    void add_gradient(size_t i, size_t j, T w, param_type& g) const {
      g(i, j) += w / a(i, j);
    }

    /**
     * Adds a gradient of the expected log-likelihood of the specified
     * data point to the gradient table g.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_gradient(const array1_type& phead, size_t j, T w,
                      param_type& g) const {
      g.col(j) += w * phead / a.col(j);
    }

      
    /**
     * Adds the diagonal of the Hessian of log-likleihood of the specified
     * data point with weight w to the Hessian diagonal h.
     * \param i the row index
     */
    void add_hessian_diag(size_t i, T w, param_type& h) const {
      h(i) -= w / (a(i) * a(i));
    }

    /**
     * Adds the diagonal of the Hessian of log-likleihood of the specified
     * data point with weight w to the Hessian diagonal h.
     * \param i the row index
     * \param j the column index
     */
    void add_hessian_diag(size_t i, size_t j, T w, param_type& h) {
      h(i, j) -= w / (a(i, j) * a(i, j));
    }
    
    /**
     * Adds the diagonal of the Hessian of the expected log-likelihoood of
     * the specified data point to the Hessian diagonal h.
     * \param phead the distribution over the row index of f
     * \param j the column index
     */
    void add_hessian_diag(const array1_type& phead, size_t j, T w,
                          param_type& h) {
      h.col(j) -= w * phead / a.col(j) / a.col(j);
    }

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

  }; // class probability_array_ll

} // namespace sill

#endif
