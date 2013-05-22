#ifndef SILL_SET_INSERT_ITERATOR_HPP
#define SILL_SET_INSERT_ITERATOR_HPP

#include <iterator>

namespace sill {

  /**
   * An output iterator that inserts elements into a set.
   *
   * @tparam Set a type which supports insertion with the insert() function
   *
   * \ingroup iterator
   */
  template <typename Set>
  class set_insert_iterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
    
  protected:
    
    /**
     * The set into which elements are inserted. 
     * \todo Could be a smart ptr (e.g. if the class is needed by users)
     */
    Set* set_ptr;
    
  public:
    
    /**
     * Constructor.  It is declared explicit to avoid it use in automatic
     * conversions.
     */
    explicit set_insert_iterator(Set& s) : set_ptr(&s) { }
    
    /**
     * Inserts an element into this iterator's backing set.
     *
     * @param e the element to insert
     */
    set_insert_iterator& operator=(const typename Set::value_type& e) {
      set_ptr->insert(e);
      return *this;
    }
    
    //! Simply returns *this.  (This iterator has no notion of position.)
    set_insert_iterator& operator*() {
      return *this;
    }
    
    //! Simply returns *this.  (This iterator does not "move".)
    set_insert_iterator& operator++() {
      return *this;
    }
    
    //! Simply returns *this.  (This iterator does not "move".)
    set_insert_iterator& operator++(int) {
      return *this;
    }
    
  };
  
  /**
   * A wrapper function that makes it easy to create a
   * set_insert_iterator object.
   * 
   * @param set the set into which elements are inserted
   * @return a set_insert_iterator object for the set
   *
   * \relates set_insert_iterator
   */
  template<typename Set>
  inline set_insert_iterator<Set> set_inserter(Set& set) {
    return set_insert_iterator<Set>(set); 
  }

} // namespace sill

#endif // SILL_SET_INSERT_ITERATOR_HPP

