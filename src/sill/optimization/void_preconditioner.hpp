
#ifndef SILL_VOID_PRECONDITIONER_HPP
#define SILL_VOID_PRECONDITIONER_HPP

namespace sill {

  /**
   * Class used when no preconditioner is used by conjugate_gradient.
   * This fits the PreconditionerFunctor concept.
   * @tparam OptVector  Type used to store optimization variables and
   *                    optimization directions.
   *
   * \ingroup optimization_classes
   */
  template <typename OptVector>
  class void_preconditioner {

  public:

    //! Applies a preconditioner to the given direction, when the optimization
    //! variables have value x.
    void precondition(OptVector& direction, const OptVector& x) const {
      return;
    }

    //! Applies the last computed preconditioner to the given direction.
    void precondition(OptVector& direction) const {
      return;
    }

  };

} // namespace sill

#endif // #ifndef SILL_VOID_PRECONDITIONER_HPP
