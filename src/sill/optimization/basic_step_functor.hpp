
#ifndef SILL_BASIC_STEP_FUNCTOR_HPP
#define SILL_BASIC_STEP_FUNCTOR_HPP

#include <sill/optimization/concepts.hpp>
#include <sill/optimization/real_opt_step_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Functor to be passed to line_search for computing the objective.
   * This fits the LineSearchObjectiveFunctor concept.
   *
   * @tparam OptVector      Datatype which stores the optimization variables.
   * @tparam Objective      Type of functor which computes the objective value.
   *
   * \ingroup optimization_classes
   */
  template <typename OptVector, typename Objective>
  struct basic_step_functor
    : public real_opt_step_functor {

    concept_assert((sill::OptimizationVector<OptVector>));
    concept_assert((sill::ObjectiveFunctor<Objective, OptVector>));

  private:

    basic_step_functor() { }

    //! Objective
    const Objective* obj_functor_ptr;

    //! Starting point x
    const OptVector* x_ptr;

    //! Direction
    const OptVector* direction_ptr;

    //! Temp place to store (x + eta * direction).
    mutable OptVector tmp_x;

  public:

    /**
     * Constructor.
     * @param x          Base x from which line search starts.
     * @param direction  Optimization direction.
     */
    basic_step_functor(const Objective& obj, const OptVector& x,
                       const OptVector& direction)
      : obj_functor_ptr(&obj), x_ptr(&x), direction_ptr(&direction),
        tmp_x(x.size(), 0.) {
    }

    //! Resets this for a new line search.
    void reset(const Objective& obj, const OptVector& x,
               const OptVector& direction) {
      obj_functor_ptr = &obj;
      x_ptr = &x;
      direction_ptr = &direction;
      if (x.size() != tmp_x.size())
        tmp_x.resize(x.size());
    }

    /**
     * Computes the value of the objective for step size eta;
     * i.e., objective(x + eta * direction).
     */
    double objective(double eta) const {
      assert(x_ptr);
      assert(direction_ptr);

      tmp_x = *direction_ptr;
      tmp_x *= eta;
      tmp_x += (*x_ptr);

      return obj_functor_ptr->objective(tmp_x);
    }

    //! Always returns false.
    bool stop_early() const {
      return false;
    }

  }; // struct basic_step_functor

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_BASIC_STEP_FUNCTOR_HPP
