#ifndef SILL_LINE_SEARCH_HPP
#define SILL_LINE_SEARCH_HPP

#include <sill/global.hpp>
#include <sill/optimization/concepts.hpp>
#include <sill/line_search/line_step_value.hpp>

#include <boost/function.hpp>

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
    typedef line_step_value<real_type> result_type;

    //! A type that represents the objective function
    typedef boost::function<real_type(const Vec&)> objective_fn;
    
    //! A type that represents the gradient of the objective
    typedef boost::function<const Vec&(const Vec&)> gradient_fn;

    //! Default constructor
    line_search()
      : bounding_steps_(0), selection_steps_(0) { }

    //! Destructor
    virtual ~line_search() { }

    //! Sets the objective and the gradient used in the search
    virtual void reset(const objective_fn& objective,
                       const gradient_fn& gradient) = 0;

    //! Compute the step in the given direction
    virtual result_type step(const Vec& x, const Vec& direction) = 0;

    //! Returns the number of bounding steps performed so far
    size_t bounding_steps() const { return bracketing_steps_; }

    //! Returns the number of selection steps performed so far
    size_t selection_steps() const { return selection_steps_; }

  protected:
    size_t bounding_steps_;
    size_t selection_steps_;

  }; // class line_search

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

