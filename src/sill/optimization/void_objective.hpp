
#ifndef SILL_VOID_OBJECTIVE_HPP
#define SILL_VOID_OBJECTIVE_HPP

#include <sill/optimization/concepts.hpp>

namespace sill {

  /**
   * Class used when no objective is used by gradient_method.
   * This fits the ObjectiveFunctor concept.
   * @tparam OptVector  Type used to store optimization variables and
   *                    optimization directions.
   *
   * \ingroup optimization_classes
   */
  template <typename OptVector>
  class void_objective {

  public:
    double objective(const OptVector& x) const {
      return std::numeric_limits<double>::max();
    }

  };

} // namespace sill

#endif // #ifndef SILL_VOID_OBJECTIVE_HPP
