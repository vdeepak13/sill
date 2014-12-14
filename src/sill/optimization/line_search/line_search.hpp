#ifndef SILL_LINE_SEARCH_HPP
#define SILL_LINE_SEARCH_HPP

#include <sill/global.hpp>
#include <sill/optimization/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An interface for algorithms that select the desired step size in
   * gradient-based optimization algorithms after the step direction
   * has been chosen.
   *
   * \ingroup optimization_algorithms
   * \tparam Vec a class that satisfies the OptimizationVector concept.
   */
  template <typename Vec>
  class line_search {
    concept_assert(OptimizationVector<Vec>)
  public:
    //! The type that represents the position and the objective value
    typedef typename Vec::value_type real_type;

    //! Default constructor
    line_search()
      : bracketing_steps_(0), selection_steps_(0) { }

    //! Destructor
    virtual ~line_search() { }

    //! Compute the step in the given direction
    virtual real_type step(const Vec& x, const Vec& direction) = 0;

    //! Returns the number of bracketing steps performed so far
    size_t bracketing_steps() const { return bracketing_steps_; }

    //! Returns the number of selection steps performed so far
    size_t selection_steps() const { return selection_steps_; }

  protected:
    size_t bracketing_steps_;
    size_t selection_steps_;

  }; // class line_search

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

