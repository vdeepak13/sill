
#ifndef PRL_VOID_STEP_FUNCTOR_HPP
#define PRL_VOID_STEP_FUNCTOR_HPP

#include <prl/optimization/real_opt_step_functor.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Class which does nothing, save for fitting the real_opt_step_functor
   * interface.
   *
   * @author Joseph Bradley
   *
   * \ingroup optimization_classes
   */
  class void_step_functor
    : public real_opt_step_functor {

    // Public methods
    //==========================================================================
  public:

    void_step_functor() { }

    //! Computes the value of the objective for step size eta.
    double objective(double eta) const {
      assert(false); // This should never be used.
      return std::numeric_limits<double>::infinity();
    }

    //! Returns true if this functor recommends early stopping (of a line
    //! search).
    bool stop_early() const {
      assert(false); // This should never be used.
      return false;
    }

  };  // class void_step_functor

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_VOID_STEP_FUNCTOR_HPP
