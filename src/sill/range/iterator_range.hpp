#ifndef SILL_ITERATOR_RANGE_HPP
#define SILL_ITERATOR_RANGE_HPP

#include <iterator>
#include <tuple>

namespace sill {

  /**
   * A class that represents a range as a pair of iterators. Besides the
   * standard begin() and end() operation, this range can be converted
   * to a tuple, so that it can be assigned to a pair of iterators via
   * std::tie.
   *
   * \tparam Iterator the underlying iterator type
   */
  template <typename Iterator>
  class iterator_range {
  public:
    //! The underlying iterator type.
    typedef Iterator iterator;

    //! The value being iterated over
    typedef typename std::iterator_traits<Iterator>::value_type value_type;

    //! Constructs a null range.
    iterator_range() { }

    //! Constructs a range with given start and end.
    iterator_range(Iterator begin, Iterator end)
      : begin_(begin), end_(end) { }

    //! Converts the range to a tuple
    operator std::tuple<Iterator&, Iterator&>() {
      return std::tuple<Iterator&, Iterator&>(begin_, end_);
    }

    //! Returns the beginning of the range.
    Iterator begin() const {
      return begin_;
    }

    //! Returns the end of the range.
    Iterator end() const {
      return end_;
    }

    //! Returns true if the range is empty.
    bool empty() const {
      return begin_ == end_;
    }

    //! Returns true if two ranges have the same begin and end.
    friend bool operator==(const iterator_range& a, const iterator_range& b) {
      return a.begin_ == b.begin_ && a.end_ == b.end_;
    }
    
    //! Returns true if tow ranges do not have the same begin or end.
    friend bool operator!=(const iterator_range& a, const iterator_range& b) {
      return !(a == b);
    }

  private:
    //! The start of the range.
    Iterator begin_;

    //! The end of the range.
    Iterator end_;
  };
} // namespace sill

#endif
