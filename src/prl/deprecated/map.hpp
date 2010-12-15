
#ifndef PRL_MAP_HPP
#define PRL_MAP_HPP

#include <sstream>
#include <stdexcept>

#include <boost/property_map.hpp>
#include <boost/concept/assert.hpp>

#include <prl/global.hpp>
#include <prl/copy_ptr.hpp>
#include <prl/set.hpp>
#include <prl/functional.hpp>
#include <prl/iterator/map_insert_iterator.hpp>
#include <prl/range/algorithm.hpp>
#include <prl/range/forward_range.hpp>
#include <prl/range/transformed.hpp>
#include <prl/serialization/map.hpp>

#include <prl/stl_concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * An implementation of a map from keys to values which can be
   * treated as a scalar type; assignments are implemented by copies
   * of reference-counted pointers.
   *
   * @tparam Key 
   *         The key type of the map.  The type must satisfy the 
   *         CopyConstructible and LessThanComparable concepts.
   * @tparam T
   *         The type of values stored in the map.  The type must
   *         satisfy the DefaultConstructible and CopyConstructible concepts.
   */
  template <typename Key, typename T>
  class map
    : public std::unary_function<const Key&, const T&>{

    concept_assert((CopyConstructible<Key>));
    concept_assert((LessThanComparable<Key>));
    concept_assert((DefaultConstructible<T>));
    concept_assert((CopyConstructible<T>));

    // Public type declarations
    //==========================================================================
  public:
    //! The type of container used to store key-value pairs.
    typedef typename std::map<Key, T> container_type;

    // Container typedefs
    typedef typename container_type::value_type      value_type;
    typedef typename container_type::difference_type difference_type;
    typedef typename container_type::size_type       size_type;
    typedef typename container_type::reference       reference;
    typedef typename container_type::pointer         pointer;
    typedef typename container_type::iterator        iterator;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::const_pointer   const_pointer;
    typedef typename container_type::const_iterator  const_iterator;

    // AssociativeContainer typedefs
    typedef typename container_type::key_type        key_type;
    typedef typename container_type::mapped_type     mapped_type;
    typedef typename container_type::key_compare     key_compare;
    typedef typename container_type::value_compare   value_compare;

    // Private data members
    //==========================================================================
  private:
    /**
     * A shared pointer to the underlying map container.  Shared pointers
     * are used so that sets can be treated as lightweight objects
     * that are copied and assigned efficiently.
     */
    copy_ptr<container_type> container_ptr;

    /**
     * Convenience function that returns a const reference to the
     * underlying container #container_ptr.
     */
    const container_type& container() const {
      return *container_ptr;
    }

    //! Returns a mutable reference to the underlying container.
    container_type& container() {
      return *container_ptr;
    }

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Creates an empty map.
    map() : container_ptr(new container_type()) { }

    //! Serializer
    void save(oarchive & ar) const {
      ar << container();
    }
    //! Deserializer
    void load(iarchive & ar) {
      ar >> container();
    }

  #ifndef SWIG
    //! Creates a map from an iterator range over key-value pairs.
    template <typename It>
    map(It begin, It end) 
      : container_ptr(new container_type(begin, end)) {
      concept_assert((InputIterator<It>));
    }

    /*
    //! Constructor; creates a map from a range of elements.
    template <typename SinglePassRange>
    explicit map(const SinglePassRange& pairs)
      : container_ptr(new container_type(boost::begin(pairs),
                                         boost::end(pairs))) { }
    */

    //! Singleton constructor
    map(const value_type& p)
      : container_ptr(new container_type(&p, &p + 1)) { }
  #endif

    //! Singleton map constructor.
    map(const Key& key, const T& value)
      : container_ptr(new container_type()) {
      this->container().insert(value_type(key, value));
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str(); 
    }

    // Accessors
    //==========================================================================
    //! Returns the size of this set.
    size_t size() const {
      return container().size();
    }

    //! implements Container::max_size()
    size_t max_size() const {
      return container().max_size();
    }

    //! Returns true iff this map is empty.
    bool empty() const {
      return container().empty();
    }

    //! Returns an iterator that points to the first key-value pair of this map.
    iterator begin() {
      return container().begin();
    }

    //! Returns an iterator that points to the first key-value pair of this map.
    const_iterator begin() const {
      return container().begin();
    }

    //! Returns a iterator that points to after the last value of this map.
    iterator end() {
      return container().end();
    }

    //! Returns a iterator that points to after the last value of this map.
    const_iterator end() const {
      return container().end();
    }

    //! Returns an iterator range over the key-value pairs of this map.
    std::pair<iterator, iterator> values() {
      return std::make_pair(container().begin(), container().end());
    }

    //! Returns an iterator range over the key-value pairs of this map.
    std::pair<const_iterator, const_iterator> values() const {
      return std::make_pair(container().begin(), container().end());
    }

    //! Returns a function object capable of comparing two keys
    key_compare key_comp() const {
      return container().key_comp();
    }

    //! Returns a function object capable of comparing the keys of two values
    value_compare value_comp() const {
      return container().value_comp();
    }

    //! Returns the number of map objects, including *this,
    //! that share the data with *this.
    long use_count() const {
      return container_ptr.use_count();
    }

    // Queries
    //==========================================================================
    /**
     * Element update.
     *
     * @param key the key whose value is to be updated
     * @return a reference to the value to be associated with the
     *         supplied key (which is initialized to a default)
     */
    T& operator[](const Key& key) {
      return container()[key];
    }

    /**
     * Element access.  This method throws a std::out_of_range
     * exception if the key has no binding in the map.
     *
     * @param key the key whose value is to be retrieved
     * @return a reference to the value to be associated with the
     *         supplied key (which is initialized to a default)
     */
    const T& operator[](const Key& key) const {
      const_iterator it = container().find(key);
      if (it != container().end())
        return it->second;
      else
        throw std::out_of_range("key not present in map");
    }

    /**
     * Element access.  This method throws a std::out_of_range
     * exception if the key has no binding in the map.  This method is
     * the same as the const operator[].
     *
     * @param key the key whose value is to be retrieved
     * @return a reference to the value to be associated with the
     *         supplied key (which is initialized to a default)
     */
    const T& get(const Key& key) const {
      return this->operator[](key);
    }

    /**
     * Element access. If the key is not present, returns the default value
     */
    const T& get(const Key& key, const T& default_value) const {
      const_iterator it = container().find(key);
      if (it != container().end())
        return it->second;
      else
        return default_value;
    }

    /**
     * Returns a const-reference to the mapped element. Functors are often
     * passed by value, so we do not provide the non-const version to avoid
     * accidental copies.
     */
    const T& operator()(const Key& key) const {
      return operator[](key);
    }

    /**
     * Element search.  This method returns an iterator to the element
     * with the supplied key, or it returns the end iterator.
     *
     * @param  key the key of the element searched for
     * @return an iterator to the element with the supplied key, or
     *         the end iterator
     */
    const_iterator find(const Key& key) const {
      return container().find(key);
    }

    //! Element search
    iterator find(const Key& key) {
      return container().find(key);
    }

    //! Element search
    size_t count(const Key& key) const {
      return container().count(key);
    }

    //! implements AssociativeContainer::equal_range
    std::pair<iterator, iterator> equal_range(const Key& key) {
      return container().equal_range(key);
    }

    //! implements AssociativeContainer::equal_range
    std::pair<const_iterator, const_iterator>
    equal_range(const Key& key) const {
      return container().equal_range(key);
    }

    //! Returns true iff the supplied key is in the map.
    bool contains(const Key& key) const {
      return (container().find(key) != container().end());
    }

    //! Returns true if the supplied keys are in this map
    bool contains(const set<Key>& keys) const {
      foreach(Key key, keys) if (!contains(key)) return false;
      return true;
    }

    //! Returns the values for a subset of the keys
    template <typename Range>
    forward_range<const T&> values(const Range& keys) const {
      return make_transformed(keys, *this);
    }

    //! Returns the set of keys of this map.
    typename prl::set<Key> keys() const {
      return prl::set<Key>(make_transformed(*this, pair_first<Key,T>()));
    }
    
    //! Returns true if this map contains any elements from the given set
    bool meets(const set<Key>& elts) const {
      return keys().meets(elts);
    }

    //! Returns the union of this map and the supplied map.
    //! If a key is present in both maps, the value from this map is copied.
    map operator+(const map& s) const {
      map u;
      prl::set_union(*this, s, map_inserter(u), value_comp());
      return u;
    }

    //! Returns the difference of this map and the supplied map.
    map operator-(const map& s) const {
      map u;
      prl::set_difference(*this, s, map_inserter(u), value_comp());
      return u;
    }

    //! Returns the intersection of this map and the supplied map.
    map intersect(const map& s) const {
      map u;
      prl::set_intersection(*this, s, map_inserter(u), value_comp());
      return u;
    }

    //! Returns a submap for the keys present in the specified set
    map intersect(const set<Key>& s) const {
      map u;
      prl::set_intersection(*this, s, map_inserter(u), key_value_compare());
      return u;
    }

    /**
     * Returns a map whose keys have been replaced according to a
     * given map.  The mapping must be unique, i.e., not change the
     * number of keys in the map.
     * @param allow_missing
     */
    template <typename Map>
    map<typename Map::mapped_type, T>
    rekey(const Map& key_map) const {
      concept_assert((UniquePairAssociativeContainer<Map>));
      concept_assert((Convertible<Key, typename Map::key_type>));
      map<typename Map::mapped_type, T> new_map;
      foreach(const value_type& v, container())
        new_map[key_map[v.first]] = v.second;

      assert(new_map.size() == size());
      return new_map;
    }

    //! Equality test.
    bool operator==(const map& s) const {
      return ((container_ptr == s.container_ptr)
              || (container() == s.container()));
    }

    //! Inequality test.
    bool operator!=(const map& s) const {
      return !this->operator==(s);
    }

    // Modifiers
    //==========================================================================
  #ifndef SWIG
    //! implements UniqueAssociativeContainer::insert
    std::pair<iterator, bool> insert(const value_type& x) {
      return container().insert(x);
    }
  #endif

    //! Inserts a key-value pair into this container
    //! \todo What if the key-value pair is already present?
    void insert(Key key, T value) {
      container().insert(std::make_pair(key, value));
    }

    //! implements UniqueAssociativeContainer::insert
    template <typename It>
    void insert(It start, It finish) {
      concept_assert((InputIterator<It>));
      container().insert(start, finish);
    }

    //! Inserts all elements of the supplied map into this set
    void insert(const map& map) {
      insert(map.begin(), map.end());
    }

    /**
     * Element removal.
     *
     * @param key the key of the element to remove
     * @return true iff the element was present in the map
     */
    bool remove(const Key& key) {
      return (container().erase(key) > 0);
    }

    //! Removes all elements in the given set
    void remove(const set<Key>& keys) {
      foreach(Key key, keys) remove(key);
    }
      
    //! implements AssociativeContainer::erase
    bool erase(const Key& key) {
      return (container().erase(key) > 0);
    }

  #ifndef SWIG
    //! implements AssociativeContainer::erase
    void erase(iterator it) {
      container().erase(it);
    }

    //! implements AssociativeContainer::erase
    void erase(iterator it, iterator end) {
      container().erase(it, end);
    }
  #endif

    //! Swaps the contents of this map with those of the supplied map.
    void swap(map& other) {
      container_ptr.swap(other.container_ptr);
    }

    //! Empties this map.
    void clear() {
      container().clear();
    }

    // Private member classes
    //==========================================================================
  private:
    struct key_value_compare 
      : public std::binary_function<value_type, Key, bool> {
      bool operator()(const Key& x, const value_type& y) const {
        return x < y.first;
      }
      bool operator()(const value_type& x, const Key& y) const {
        return x.first < y;
      }
      // we also need to support key-key and value-value comparisons
      // because MSVC checks if the ranges are ordered
      bool operator()(const Key& x, const Key& y) const {
        return x < y;
      }
      bool operator()(const value_type& x, const value_type& y) const {
        return x.first < y.first;
      }
    }; // struct key_value_compare

  };

  //! Writes a map to the supplied stream.
  //! \relates map
  template <typename Key, typename T>
  std::ostream& operator<<(std::ostream& out, const map<Key, T>& m) {
    out << "{";
    for (typename map<Key, T>::const_iterator it = m.begin(); it != m.end();) {
      out << it->first << "-->" << it->second;
      if (++it != m.end()) out << " ";
    }
    out << "}";
    return out;
  }

  //! Property map interface
  //! \relates map
  template <typename Key, typename T>
  const T& get(const prl::map<Key, T>& map, Key key) {
    return map[key];
  }

  //! Creates an identity map (a map from elements to themselves)
  //! \relates map
  template <typename Key>
  map<Key, Key> make_identity_map(const set<Key>& keys) {
    map<Key, Key> m;
    foreach(const Key& key, keys) 
      m[key] = key;
    return m;
  }

} // namespace prl


namespace boost
{
  //! prl::map can also act as a readable property map
  template <typename Key, typename T>
  struct property_traits<prl::map<Key, T> > {
    typedef prl::map<Key, T> container;
    typedef typename container::key_type key_type;
    typedef typename container::mapped_type value_type;
    typedef const typename container::mapped_type& reference;
    typedef boost::readable_property_map_tag category;
  };

} // namespace boost

#include <prl/macros_undef.hpp>


#endif // #ifndef PRL_MAP_HPP
