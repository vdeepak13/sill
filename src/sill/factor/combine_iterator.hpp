
#ifndef SILL_COMBINE_ITERATOR_HPP
#define SILL_COMBINE_ITERATOR_HPP

#include <iterator>

#include <sill/global.hpp>
#include <sill/factor/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An output iterator that combines together the factors that are
   * assigned to it.
   *
   * Note that copies of combine_iterator do not hold a shared state.
   * Therefore, if a combine_iterator is passed by value to a function
   * (which is the accepted standard), the function needs to return
   * the resulting combine_iterator back to the caller.
   *
   * @tparam F a class that models the Factor concept
   * @tparam Op a class that combines factors, e.g., sill::inplace_multiplies
   *
   * \todo For some factors, the iterator could be specialized to
   *       obtain a more efficient implementation. For example,
   *       for canonical Gaussian factors, the memory could be 
   *       allocated for the union of all argument to avoid reallocations
   *       when introducing new variables in the combination.
   *
   * \ingroup factor_types, iterator
   */
  template <typename F, typename Op>
  class combine_iterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
    concept_assert((Factor<F>));

  private:
    //! The combination of all factors previously assigned to this iterator.
    F combination;

    //! The combination operation
    Op op;

  public:
    //! Creates the iterator with the specified initial value and operation
    combine_iterator(const F& init, Op op)
      : combination(init), op(op) { }

    //! Adds the factor to the combination
    combine_iterator& operator=(const F& factor) {
      op(combination, factor);
      return *this;
    }

    //! Simply returns *this.  (This iterator has no notion of position.)
    combine_iterator& operator*() {
      return *this;
    }

    //! Advances to the next "position" (noop)
    const combine_iterator& operator++() {
      return *this;
    }

    //! Advances to the next "position" (noop)
    combine_iterator operator++(int) {
      return *this;
    }

    //! Retrieves the combination of all assigned factors.
    const F& result() const {
      return combination;
    }

  }; // class combine_iterator

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_COMBINE_ITERATOR_HPP
