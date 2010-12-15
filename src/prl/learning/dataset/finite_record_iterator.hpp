
#ifndef PRL_FINITE_RECORD_ITERATOR_HPP
#define PRL_FINITE_RECORD_ITERATOR_HPP

#include <iterator>

#include <prl/learning/dataset/finite_record.hpp>

namespace prl {

  // Pre-declaration
  class finite_record;

  /**
   * Iterator for values in a finite_record.
   * This is meant to help bring records closer to a replacement for maps.
   * NOTE: This iterates over record values in the order defined by
   *       the finite_numbering_ptr, not in the order of the value vector.
   */
  class finite_record_iterator
    : public std::iterator<std::forward_iterator_tag,
                           const std::pair<finite_variable*, size_t> > {

    // Private data and methods
    //==========================================================================

    const finite_record* r_ptr;

    //! Iterator into r_ptr->finite_numbering_ptr for current value.
    std::map<finite_variable*, size_t>::const_iterator iter;

    //! Current value.
    mutable std::pair<finite_variable*, size_t> val;

    //! Set val to the current value for iter.
    void set_val() const;

    // Public methods
    //==========================================================================
  public:

    /**
     * Constructor.
     * @param r  Finite record.
     * @param v  Variable to which to point this iterator.
     *           If NULL or not in record r, this is an end iterator.
     */
    finite_record_iterator
    (const finite_record& r, finite_variable* v);

    //! Prefix increment.
    finite_record_iterator& operator++();

    //! Postfix increment.
    finite_record_iterator operator++(int);

    //! Returns a const reference to the current <variable, value> pair.
    const std::pair<finite_variable*, size_t>& operator*() const;

    //! Returns a const pointer to the current <variable, value> pair.
    const std::pair<finite_variable*, size_t>* const operator->() const;

    //! Returns truth if the two iterators are the same.
    bool operator==(const finite_record_iterator& it) const;

    //! Returns truth if the two iterators are different.
    bool operator!=(const finite_record_iterator& it) const;

  }; // class finite_record_iterator

} // namespace prl

#endif // #ifndef PRL_FINITE_RECORD_ITERATOR_HPP
