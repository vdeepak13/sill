#ifndef SILL_INTEGRATION_POINTS_APPROXIMATOR_HPP
#define SILL_INTEGRATION_POINTS_APPROXIMATOR_HPP

#include <utility> // for std::pair

#include <sill/factor/approx/interfaces.hpp>
#include <sill/math/matrix.hpp>
#include <sill/math/vector.hpp>

namespace sill {

  /**
   * A Gaussian approximator that uses a set of fixed integation points.
   * For now, the approximator uses a set of Exact Monomials degree 5 points.
   * \ingroup factor_approx
   */
  class integration_points_approximator : public gaussian_approximator {

  public:
    integration_points_approximator* clone() const {
      return new integration_points_approximator(*this);
    }

    moment_gaussian operator()(const nonlinear_gaussian& ng,
                               const moment_gaussian& prior) const;

    //! Returns a set of points and the associated weights
    //! The points are guaranteed to be centered
    static std::pair<mat, vec> points(int d);

  };

} // namespace sill

#endif
