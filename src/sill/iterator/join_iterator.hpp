#ifndef SILL_JOIN_ITERATOR_HPP
#define SILL_JOIN_ITERATOR_HPP

#include <iterator>

namespace sill {
  
  /**
   * An iterator that merges two 
   * \ingroup iterator
   */
  template <typename It1, typename It2>
  class join_iterator 
    : public std::iterator<std::forward_iterator_tag,
                           typename std::iterator_traits<It1>::value_type,
                           typename std::iterator_traits<It1>::difference_type,
                           typename std::iterator_traits<It1>::pointer,
                           typename std::iterator_traits<It1>::reference> {
    
  private:
    //! The current iterator and the end of the first iteration range
    It1 it1, end1;

    //! The current iterator of the second iteration range
    It2 it2;
    
  public:
    join_iterator() : it1(), end1(), it2() { }

    join_iterator(It1 it1, It1 end1, It2 it2) 
      : it1(it1), end1(end1), it2(it2) { }
    
    typename std::iterator_traits<It1>::reference operator*() const { 
      if (it1 == end1)
        return *it2;
      else
        return *it1;
    }
    
    join_iterator& operator++() {
      if (it1 == end1) 
        ++it2;
      else
        ++it1;
      return *this;
    }

    join_iterator operator++(int) {
      join_iterator tmp(*this);
      ++*this;
      return tmp;
    }

    bool operator==(const join_iterator& other) const {
      return it1 == other.it1 && it2 == other.it2;
    }

    bool operator!=(const join_iterator& other) const {
      return !operator==(other);
    }

  }; // class join_iterator


  //! \relates join_iterator
  template <typename It1, typename It2>
  join_iterator<It1, It2>
  make_join_iterator(const It1& it1, const It1& end1, const It2& it2) {
    return join_iterator<It1, It2>(it1, end1, it2);
  }

} // namespace sill

#endif
