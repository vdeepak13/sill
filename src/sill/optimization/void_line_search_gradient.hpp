#ifndef SILL_VOID_LINE_SEARCH_GRADIENT_HPP
#define SILL_VOID_LINE_SEARCH_GRADIENT_HPP

namespace sill {

  /**
   * Class used when no gradient is used by line_search.
   * This fits the LineSearchGradientFunctor concept.
   *
   * \ingroup optimization_classes
   */
  class void_line_search_gradient {

  public:

    //! Computes the gradient of the objective (w.r.t. eta) for step size eta.
    //! This should never really be called!
    double gradient(double eta) const {
      assert(false);
      return - std::numeric_limits<double>::infinity();
    }

  };

} // namespace sill

#endif // #ifndef SILL_VOID_LINE_SEARCH_GRADIENT_HPP
