#ifndef SILL_LINE_SEARCH_HPP
#define SILL_LINE_SEARCH_HPP

#include <sill/global.hpp>
#include <sill/optimization/concepts.hpp>
#include <sill/optimization/gradient_objective.hpp>
#include <sill/optimization/line_search/line_search_result.hpp>

#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An interface for algorithms that select the desired step size in
   * gradient-based optimization algorithms after the step direction
   * has been chosen.
   *
   * \ingroup optimization_algorithms
   *
   * \tparam Vec a class that satisfies the OptimizationVector concept.
   */
  template <typename Vec>
  class line_search {
    concept_assert(OptimizationVector<Vec>)
  public:
    //! The storage type of the vector
    typedef typename Vec::value_type real_type;

    //! A type that represents the step and the corresponding objective value
    typedef line_search_result<real_type> result_type;

    //! Default constructor
    line_search()
      : bounding_steps_(0), selection_steps_(0) { }

    //! Destructor
    virtual ~line_search() { }

    /**
     * Sets the objective used in the search.
     * The pointer is not owned by this object.
     */
    virtual void objective(gradient_objective<Vec>* objective) = 0;

    /**
     * Computes the step in the given direction.
     * Returns the step size and the corresponding objective value.
     */
    virtual result_type step(const Vec& x, const Vec& direction) = 0;

    /**
     * Prints the line search algorithm and its parameters to a stream.
     */
    virtual void print(std::ostream& out) const = 0;

    /**
     * Returns the number of bounding steps performed so far.
     * These are the steps in algorithms, such as bracketing line
     * search, where a valid range for the step size must be
     * initially computed.
     */
    size_t bounding_steps() const { return bounding_steps_; }

    /**
     * Returns the number of selection steps performed so far.
     * These are the steps to narrow down the initial estimate
     * into an acceptable value.
     */
    size_t selection_steps() const { return selection_steps_; }

  protected:
    size_t bounding_steps_;
    size_t selection_steps_;

  }; // class line_search


  /**
   * Prints the line_search object to a stream.
   * \relates line_search
   */
  template <typename Vec>
  std::ostream& operator<<(std::ostream& out, const line_search<Vec>& ls) {
    ls.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

