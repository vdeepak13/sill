#ifndef SILL_MAP_INSERT_ITERATOR_HPP
#define SILL_MAP_INSERT_ITERATOR_HPP

#include <iterator>

namespace sill {

  /**
   * An output iterator that inserts elements into a map.
   *
   * @tparam Map a type which supports insertion with operator[]
   *
   * \ingroup iterator
   */
  template <typename Map>
  class map_insert_iterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {

  protected:

    //! The map into which elements are inserted.
    Map* map_ptr;

  public:

    //! Constructor (declared explicit to avoid automatic conversions).
    explicit map_insert_iterator(Map& m) : map_ptr(&m) { }

    /**
     * Inserts an element into this iterator's backing map.
     *
     * @param e the element to insert
     */
    map_insert_iterator& operator=(const typename Map::value_type& e) {
      (*map_ptr)[e.first] = e.second;
      return *this;
    }

    //! Simply returns *this.  (This iterator has no notion of position.)
    map_insert_iterator& operator*() {
      return *this;
    }

    //! Simply returns *this.  (This iterator does not "move".)
    map_insert_iterator& operator++() {
      return *this;
    }

    //! Simply returns *this.  (This iterator does not "move".)
    map_insert_iterator& operator++(int) {
      return *this;
    }

  };

  /**
   * A wrapper function that makes it easy to create a
   * map_insert_iterator object.
   * 
   * @param map the map into which elements are inserted
   * @return a map_insert_iterator object for the map
   *
   * \todo Stano: looks similar to the STL insert_iterator
   * \relates map_insert_iterator
   */
  template<typename Map>
  inline map_insert_iterator<Map> map_inserter(Map& map) {
    return map_insert_iterator<Map>(map); 
  }

} // namespace sill

#endif // SILL_MAP_INSERT_ITERATOR_HPP
