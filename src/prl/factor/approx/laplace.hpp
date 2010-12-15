#ifndef PRL_LAPLACE_APPROXIMATOR_HPP
#define PRL_LAPLACE_APPROXIMATOR_HPP

#include <prl/factor/approx/interfaces.hpp>

namespace prl {

  /** 
   * Laplace Gaussian approximator.
   * \ingroup factor_approx
   */
  template <typename LA>
  class laplace_approximator : public gaussian_approximator<LA> {
  public:
    typedef typename LA::value_type value_type;
    typedef typename LA::vector_type vector_type;
    typedef typename LA::matrix_type matrix_type;
    typedef typename LA::index_range index_range;

    laplace_approximator* clone() const {
      return new laplace_approximator(*this);
    }

    moment_gaussian<LA> operator()(const nonlinear_gaussian<LA>& ng,
                                   const moment_gaussian<LA>& prior) const {
      assert(prior.marginal() && prior.size() == ng.size_tail());
      // Find the mode of the joint distribution using gradient descent(for now)
      
    }

  }; // class laplace_approximator

} // namespace prl

#endif
