
#ifndef SILL_REAL_OPTIMIZER_HPP
#define SILL_REAL_OPTIMIZER_HPP

#include <sill/optimization/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Interface for real-valued optimization algorithms which are iterative.
   * These algorithms are for unconstrained nonlinear optimization,
   * and they try to MINIMIZE the objective.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_algorithms
   */
  template <typename OptVector>
  class real_optimizer {

    concept_assert((sill::OptimizationVector<OptVector>));

    // Protected data
    //==========================================================================
  protected:

    //! Current values of variables being optimized over.
    OptVector& x_;

    //! Last change in objective value.
    double objective_change_;

    //! Current objective value.
    double objective_;

    //! Iteration number (which is also the number of line searches done).
    size_t iteration_;

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor.
     * @param x_          Pre-allocated and initialized variables being
     *                    optimized over.
     * @param objective_  Initial objective value.
     */
    real_optimizer(OptVector& x_, double objective_)
      : x_(x_), objective_change_(std::numeric_limits<double>::infinity()),
        objective_(objective_), iteration_(0) { }

    virtual ~real_optimizer() { }

    //! Perform one step.
    //! @return  False if converged.
    virtual bool step() = 0;

    //! Current values of variables being optimized over.
    const OptVector& x() const {
      return x_;
    }

    //! Last change in objective value.
    double objective_change() const {
      return objective_change_;
    }

    //! Current objective value.
    double objective() const {
      return objective_;
    }

    //! Current iteration (from 0), i.e., number of iterations completed.
    size_t iteration() const {
      return iteration_;
    }

  }; // class real_optimizer

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_REAL_OPTIMIZER_HPP
