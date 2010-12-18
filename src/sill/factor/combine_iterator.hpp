
#ifndef SILL_COMBINE_ITERATOR_HPP
#define SILL_COMBINE_ITERATOR_HPP

#include <iterator>

#include <sill/global.hpp>
#include <sill/copy_ptr.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/math/gdl_enum.hpp>

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
   * @tparam F a class that models that F concept
   *
   * \todo For some factors, the iterator could be specialized to
   *       obtain a more efficient implementation. For example,
   *       for canonical Gaussian factors, the memory could be 
   *       allocated for the union of all argument to avoid reallocations
   *       when introducing new variables in the combination.
   *
   * \ingroup factor_types, iterator
   */
  template <typename F>
  class combine_iterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
    concept_assert((Factor<F>));

  protected:

    //! The combination operation
    op_type op;

    //! The combination of all factors previously assigned to this iterator.
    F combination;

  public:

    //! Creates an Constructor.
    combine_iterator(op_type op)
      : op(op) { 
      switch(op) {
        case sum_op:
          combination = 0.0; break;
        case minus_op:
          /* Not well defined */
          combination = 0.0; break;
        case product_op:
          combination = 1.0; break;
        case divides_op:
          /* Not well defined */
          combination = 1.0; break;
        case max_op:
          combination = -std::numeric_limits<double>::infinity(); break;
        case min_op:
          combination = std::numeric_limits<double>::infinity();  break;
        case and_op:
          combination = 1.0; break;
        case or_op:
          combination = 0.0; break;
        default:
          assert(false); /* Should never reach here */
      }
    }

    //! Assignment.
    combine_iterator& operator=(const F& factor) {
      //! \todo At the moment, this may fail if combination does not
      //!       include all the arguments of factor.
      combination.combine_in(factor, op);
      return *this;
    }

    //! Simply returns *this.  (This iterator has no notion of position.)
    combine_iterator& operator*() {
      return *this;
    }

    //! Advances to the next "position".
    const combine_iterator& operator++() {
      return *this;
    }

    //! Advances to the next "position".
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
