
#ifndef SILL_COUNTING_OUTPUT_ITERATOR_HPP
#define SILL_COUNTING_OUTPUT_ITERATOR_HPP

#include <iterator>

#include <sill/global.hpp>

namespace sill {

  /**
   * An output iterator that counts how many times a value has been stored.
   * Similarly to insert iterators in STL, operator* returns itself, so that
   * operator= can increase the counter.
   * \ingroup iterator
   */
  class counting_output_iterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {

  private:
    //! The counter.
    size_t counter;

  public:
    //! Constructor. 
    counting_output_iterator() : counter() {}

    //! Increment the counter.
    template <typename T>
    counting_output_iterator& operator=(const T&) {
      counter++;
      return *this;
    }

    //! Dereferences to en empty target object that can absorb any assignment.
    counting_output_iterator& operator*() {
      return *this;
    }

    //! Returns *this with no side-effects.
    counting_output_iterator& operator++() {
      return *this;
    }

    //! Returns *this with no side-effects.
    counting_output_iterator& operator++(int) {
      return *this;
    }

    //! Returns the number of positions that have been assigned.
    size_t count() const {
      return counter;
    }

  }; // class counting_output_iterator

} // namespace sill

#endif // #ifndef SILL_COUNTING_OUTPUT_ITERATOR_HPP
