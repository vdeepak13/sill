#ifndef SILL_SOFTMAX_MLE_HPP
#define SILL_SOFTMAX_MLE_HPP

#include <sill/math/likelihood/softmax_ll.hpp>
#include <sill/optimization/gradient_objective.hpp>
#include <sill/optimization/gradient_method/conjugate_gradient.hpp>
#include <sill/optimization/gradient_method/gradient_descent.hpp>
#include <sill/optimization/line_search/backtracking_line_search.hpp>
#include <sill/optimization/line_search/slope_binary_search.hpp>

namespace sill {

  /**
   * A maximum likelihood estimator for the softmax parameters.
   * The maximum likelihood estimate is computed iteratively
   * using conjugate gradient descent.
   *
   * \tparam T the real type representing the parameters
   */
  template <typename T>
  class softmax_mle {
  public:
    //! The regularization parameter
    typedef T regul_type;
    
    //! The parameters returned by this estimator.
    typedef softmax_param<T> param_type;

    /**
     * Creates a maximum likelihood estimator with the specified
     * regularization parameters.
     */
    explicit softmax_mle(T regul = T(),
                         size_t max_iter = 1000,
                         bool verbose = false)
      : regul_(regul), max_iter_(max_iter), verbose_(verbose) { }

    /**
     * Computes the maximum likelihood estimate of a softmax distribution
     * using the samples in the given range. The finite portion of each
     * record represents the label, while the vector represents the features.
     * The softmax parameter structure must be preallocated to the correct
     * size, but does not need to be initialized to any particular values.
     *
     * \tparam Range a range with values convertible to
     *         std::pair<hybrid_index<T>, T>
     */
    template <typename Range>
    void estimate(const Range& samples, softmax_param<T>& p) const {
      p.fill(T(0));
      auto search = new slope_binary_search<param_type>(
        1e-6,
        wolfe_conditions<T>::param_type::conjugate_gradient()
      );
      typename conjugate_gradient<param_type>::param_type cg_param;
      cg_param.precondition = false;//true;
      conjugate_gradient<param_type> optimizer(search, cg_param);
      softmax_objective<Range> objective(samples, regul_);
      optimizer.objective(&objective);
      optimizer.solution(p);
      for (size_t it = 0; !optimizer.converged() && it < max_iter_; ++it) {
        line_search_result<T> value = optimizer.iterate();
        if (verbose_) {
          std::cout << "Iteration " << it << ", " << value << std::endl;
        }
      }
      if (!optimizer.converged()) {
        std::cerr << "Warning: failed to converge" << std::endl;
      }
      if (verbose_) {
        std::cout << "Number of calls: "
                  << objective.value_calls << " "
                  << objective.grad_calls << std::endl;
      }
      p = optimizer.solution();
    }

  private:
    //! Regularization parameters
    regul_type regul_;

    //! The maximum number of iterations
    size_t max_iter_;

    //! Set true for a verbose output
    bool verbose_;

    /**
     * A class that iterates over the dataset, computing the value and
     * the derivatives of the softmax log-likelihood function.
     */
    template <typename Range>
    struct softmax_objective : public gradient_objective<softmax_param<T> > {
      softmax_objective(const Range& samples, regul_type regul)
        : samples(samples), regul(regul), value_calls(0), grad_calls(0) { }

      T value(const softmax_param<T>& x) {
        softmax_ll<T> f(x);
        T result = T(0);
        T weight = T(0);
        for (const auto& r : samples) {
          result += f.log(r.first) * r.second;
          weight += r.second;
        }
        result /= -weight;
        result += 0.5 * regul * dot(x, x);
        ++value_calls;
        return result;
      }

      const softmax_param<T>& gradient(const softmax_param<T>& x) {
        softmax_ll<T> f(x);
        g.zero(x.labels(), x.features());
        T weight = T(0);
        for (const auto& r : samples) {
          f.add_gradient(r.first, r.second, g);
          weight += r.second;
        }
        g /= -weight;
        update(g, x, regul);
        ++grad_calls;
        return g;
      }

      const softmax_param<T>& hessian_diag(const softmax_param<T>& x) {
        softmax_ll<T> f(x);
        h.zero(x.labels(), x.features());
        T weight = T(0);
        for (const auto& r : samples) {
          f.add_hessian_diag(r.first, r.second, h);
          weight += r.second;
        }
        h /= -weight;
        h += regul;
        return h;
      }

      const Range& samples;
      T regul;
      param_type g;
      param_type h;
      size_t value_calls;
      size_t grad_calls;

    }; // struct softmax_objective

  }; // class softmax_mle

} // namespace sill

#endif
