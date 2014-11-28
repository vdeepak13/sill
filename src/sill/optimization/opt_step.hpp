#ifndef SILL_OPT_STEP_HPP
#define SILL_OPT_STEP_HPP

#include <sill/optimization/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Interface for algorithms which execute steps in gradient-based optimization
   * algorithms (after the step direction has been chosen).
   *
   * \ingroup optimization_algorithms
   * \tparam Vec a class that satisfies the OptimizationVector concept.
   */
  template <typename Vec>
  class opt_step {
    concept_assert(OptimizationVector<Vec>)
  public:
    typedef typename Vec::value_type real_type;

    real_opt_step() { }

    virtual ~real_opt_step() { }

    /**
     * Apply step in the given direction.
     * \return resulting objective value
     */
    virtual real_type apply(Vec& x, const Vec& direction) = 0;

  }; // class opt_step

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

