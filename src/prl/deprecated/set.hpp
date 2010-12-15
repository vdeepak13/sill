
#ifndef PRL_SET_HPP
#define PRL_SET_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <set>

#include <boost/concept/assert.hpp>
#include <boost/next_prior.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>

#include <prl/global.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/serialization/set.hpp>
#include <prl/copy_ptr.hpp>
#include <prl/range/algorithm.hpp>
#include <prl/iterator/set_insert_iterator.hpp>
#include <prl/iterator/counting_output_iterator.hpp>
#include <prl/stl_io.hpp>

#include <prl/stl_concepts.hpp>
#include <prl/range/concepts.hpp>

#include <prl/range/algorithm.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * An implementation of a set which can be treated as a scalar type;
   * assignments are implemented by copies of reference-counted
   * pointers.
   *
   * @tparam T
   *         The type of elements stored in this set.  The type must
   *         satisfy the CopyConstructible and LessThanComparable concepts.
   *
   * \todo Convert some of the member functions to free functions?
   */
  template <typename T>
  class set {
    concept_assert((CopyConstructible<T>));
    concept_assert((LessThanComparable<T>));

    // Public type declarations
    //==========================================================================
  public:
    typedef std::set<T> container_type;

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
    typedef typename container_type::key_compare     key_compare;
    typedef typename container_type::value_compare   value_compare;

    // Data members
    //==========================================================================
  private:
    /**
     * A shared copy-on-write pointer to the underlying container of
     * set elements.  Shared pointers are used so that sets can be
     * treated as lightweight objects that are copied and assigned
     * efficiently.
     */
    copy_ptr<container_type> container_ptr;



    //! Returns a mutable reference to the underlying container
    container_type& container() {
      return *container_ptr;
    }

    //! Returns a mutable reference to the underlying container
    const container_type& container() const{
      return *container_ptr;
    }

  public:
    //! The empty-set.
    static const set empty_set;

    //! Serializer
    void save(oarchive & ar) const {
      ar << container();
    }
    //! Deserializer
    void load(iarchive & ar) {
      ar >> container();
    }

    // Constructors and conversion operators
    //==========================================================================
  public:
    //! Default constructor; creates an empty set.
    set() : container_ptr(empty_set.container_ptr) { }

    //! Singleton set constructor.
    set(const T& elt)
      : container_ptr(new container_type(&elt, &elt + 1)) { }

    //! Constructor; creates a set from an iterator range.
    #ifndef SWIG
    template <typename It>
    set(It begin, It end) {
      concept_assert((InputIterator<It>));
      //concept_assert((Convertible<typename iterator_value<It>::type,T>));
      container_ptr.reset(new container_type(begin, end));
    }

    //! Constructor; creates a set from a range @see Boost.Range
    template <typename Range>
    explicit set(const Range& values) {
      concept_assert((InputRangeConvertible<Range,T>));
      container_ptr.reset(new container_type(boost::begin(values),
                                             boost::end(values)));
    }
    #endif

    //! Conversion from an std::vector of values (needed by SWIG)
    set(const std::vector<T>& values)
      : container_ptr(new container_type(values.begin(), values.end())) { }

    //! Value-type conversion
    template <typename U>
    set(const set<U>& other,
        typename boost::enable_if< boost::is_convertible<U, T> >::type* = 0)
      : container_ptr(new container_type(other.begin(), other.end())) {
      static_assert((!boost::is_same<T, U>::value));
      // make sure we are not performing conversion when we do not need to
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

    //! Swaps the contents of this set with those of the supplied set.
    void swap(set& other) {
      container_ptr.swap(other.container_ptr);
    }

    // Accessors
    //==========================================================================
    //! Returns the size of this set.
    size_t size() const {
      return container().size();
    }

    //! Returns true iff this set is empty.
    bool empty() const {
      return container().empty();
    }

    /**
     * Returns an iterator range over the elements of this set.
     * \deprecated use range-based iteration instead
     */
    std::pair<const_iterator, const_iterator> values() const {
      return std::make_pair(container().begin(), container().end());
    }

    //! Returns an iterator that points to the first element of this set.
    const_iterator begin() const {
      return container().begin();
    }

    //! Returns an iterator that points to the first element of this set.
    iterator begin() {
      return container().begin();
    }

    //! Returns an iterator that points to after the last element of this set.
    const_iterator end() const {
      return container().end();
    }

    //! Returns an iterator that points to after the last element of this set.
    iterator end() {
      return container().end();
    }

    //! Returns the number of set objects, *this included, that share the
    //! data with *this.
    long use_count() const {
      return container_ptr.use_count();
    }

    // Queries
    //==========================================================================
    //! Returns true if the supplied element is a member of this set.
    bool contains(const T& e) const {
      return (container().find(e) != container().end());
    }

    //! Returns the i-th element of the set in the natural ordering.
    //! This function has linear time complexity.
    const_reference operator[](size_t i) const {
      return *boost::next(begin(), i);
    }

    //! Equality test.
    bool operator==(const set& s) const {
      return ((container_ptr == s.container_ptr)
              || (container() == s.container()));
    }

    //! Inequality test.
    bool operator!=(const set& s) const {
      return !operator==(s);
    }

    //! Lexicographic less-than.
    bool operator<(const set& s) const {
      return ((container_ptr != s.container_ptr)
              && (container() < s.container()));
    }

    //! Lexicographic less-than-or-equal-to.
    bool operator<=(const set& s) const {
      return ((container_ptr == s.container_ptr)
              || (container() <= s.container()));
    }

    //! Lexicographic greater-than.
    bool operator>(const set& s) const {
      return ((container_ptr != s.container_ptr)
              && (container() > s.container()));
    }

    //! Lexicographic greater-than-or-equal-to.
    bool operator>=(const set& s) const {
      return ((container_ptr == s.container_ptr)
              || (container() >= s.container()));
    }

    //! Returns true iff every member of this set is also a member of s.
    //! \todo use std::includes or prl::includes
    bool subset_of(const set& s) const {
      if (container_ptr == s.container_ptr) return true;
      //! \todo This can be made more efficient by using iterator hints.
      foreach(T v, container()) {
        if (!s.contains(v)) return false;
      }
      return true;
    }

  #ifndef SWIG
    //! A functor that tests if this set is a subset of a set
    struct is_subset_functor : std::unary_function<set, bool> {
      set s;
      is_subset_functor(const set& s) : s(s) { }
      bool operator()(const set& t) const { return s.subset_of(t); }
    };

    is_subset_functor subset_of() const {
      return is_subset_functor(*this);
    }
  #endif

    //! Returns true iff every member of s is also a member of this set.
    bool superset_of(const set& s) const {
      return s.subset_of(*this);
    }

    //! Returns the union of this set and the supplied set.
    set plus(const set& s) const {
      set u;
      prl::set_union(*this, s, set_inserter(u));
      return u;
    }

    //! Returns the union of this set and the supplied element.
    set plus(const T& e) const {
      set u(*this);
      u.insert(e);
      return u;
    }

    //! Returns the number of elements in the union of two sets
    size_t union_size(const set& s) const {
      counting_output_iterator counter;
      return prl::set_union(*this, s, counter).count();
    }

    //! Returns the union of this set and the supplied set.
    set operator+(const set& s) const {
      return plus(s);
    }

    //! Returns the set difference of this set and the supplied set.
    set minus(const set& s) const {
      set u;
      prl::set_difference(*this, s, set_inserter(u));
      return u;
    }

    //! Returns the set difference of this set and the supplied element.
    set minus(const T& e) const {
      set u(*this);
      u.remove(e);
      return u;
    }

    //! Returns the set difference of this set and the supplied set.
    set operator-(const set& s) const {
      return minus(s);
    }

    //! Returns the intersection of this set and the supplied set.
    set intersect(const set& s) const {
      set u;
      prl::set_intersection(*this, s, set_inserter(u));
      return u;
    }

    //! Returns the number of elements shared by this set and the supplied set.
    size_t intersection_size(const set& s) const {
      counting_output_iterator counter;
      return prl::set_intersection(*this, s, counter).count();
    }

    //! Returns true if this set and the supplied set are disjoint.
    bool disjoint_from(const set& s) const {
      return (this->intersection_size(s) == 0);
    }

    //! Returns true if this set and the supplied set are not disjoint.
    bool meets(const set& s) const {
      return (this->intersection_size(s) > 0);
    }

    //! Returns a representative element of this set
    T representative() const {
      assert(!empty());
      return *begin();
    }

    /**
     * Partitions the supplied set into two sets: one is a subset of
     * this set, and the other is disjoint from this set.
     *
     * @param  s a set
     * @return a pair of sets whose union is s; the first is a subset
     *         of this set; the other is disjoint from this set.
     *         I.e., this function returns
     *         std::make_pair(this->intersect(s), s.minus(*this)).
     */
    std::pair<set, set> partition(const set& s) const {
      set a, b;
      // Calling container() ensures that we have unique ownership.
      // Hence, the container will not be reallocated, and the iterators
      // remain valid.
      iterator a_it = a.container().begin();
      iterator b_it = b.container().begin();
      const_iterator r_it, r_end, s_it, s_end;
      boost::tie(r_it, r_end) = values();
      boost::tie(s_it, s_end) = s.values();
      while ((r_it != r_end) && (s_it != s_end)) {
        if (*r_it == *s_it) {
          a_it = a.container().insert(a_it, *r_it);
          ++r_it;
          ++s_it;
        } else if (*r_it < *s_it) {
          ++r_it;
        } else {
          b_it = b.container().insert(b_it, *s_it);
          ++s_it;
        }
      }
      b.container().insert(s_it, s_end);
      return std::make_pair(a, b);
    }

    /**
     * Partitions this set into two sets: one is a subset of
     * the supplied set, and the other is disjoint from that set.
     */
    std::pair<set,set> partition_by(const set& s) const {
      return s.partition(*this);
    }

    /**
     * Returns true iff the supplied element is the least common
     * element of the two sets.  If they have no common element then
     * this function returns false.
     */
    static bool is_lce(const T& e, const set& r, const set& s) {
      const_iterator r_it, r_end, s_it, s_end;
      boost::tie(r_it, r_end) = r.values();
      boost::tie(s_it, s_end) = s.values();
      while ((r_it != r_end) && (s_it != s_end)) {
        if (*r_it == *s_it) {
          return (*r_it == e);
        } else if (e < *r_it) {
          return false;
        } else if (e < *s_it) {
          return false;
        } else if (*r_it < *s_it) {
          ++r_it;
        } else {
          ++s_it;
        }
      }
      return false;
    }

    // Modifiers
    //==========================================================================
    //! Inserts an element into this set.
    //! @return true iff the element was not already present in the set
    bool insert(T elt) {
      return container().insert(elt).second;
    }

    //! Inserts all elements of the supplied set into this set.
    //! @param set the set of elements to insert
    void insert(const set& set) {
      container().insert(set.container().begin(),
                         set.container().end());
    }

    //! Inserts all elements (or sets of elements) from a collection
    template <typename R>
    void insert(const R& elements,
                typename R::value_type = typename R::value_type()) {
      concept_assert((InputRange<R>));
      foreach(typename R::value_type elt, elements) insert(elt);
    }

    //! Inserts an element into this set.
    set& operator+=(T elt) {
      insert(elt); return *this;
    }

    //! Inserts all elements of the supplied set into this set.
    set& operator+=(const set& set) {
      insert(set); return *this;
    }

    //! Removes an element from this set
    //! @return true iff the element was present in the set
    bool remove(T elt) {
      return (container().erase(elt) > 0);
    }

    //! Removes all elements in the supplied set from this set.
    void remove(const set& s) {
      if (container_ptr == s.container_ptr)
        container().clear();
      else
        foreach(T v, s) container().erase(v);
    }

    //! Empties this set.
    void clear() {
      container().clear();
    }

  }; // class set

  //! Declaration of the empty-set.
  template <typename T>
  const set<T> set<T>::empty_set = set<T>(std::set<T>());
  // here, we must not use the default constructor since
  // the default constructor initializes to empty_set

  // Free functions
  //============================================================================

  //! Writes a human representation of the set to the supplied stream.
  //! \relates set
  template <typename Char, typename Traits, typename T>
  std::basic_ostream<Char, Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out, const set<T>& s) {
    return print_range(out, s, '{', ' ', '}');
  }

  //! Computes the union of a collection of sets
  //! \relates set
  template <typename Range>
  typename Range::value_type set_union(const Range& sets) {
    typedef typename Range::value_type set_type;
    set_type result;
    foreach(const set_type& set, sets) result += set;
    return result;
  }

  //! Computes the intersection of two sets
  //! \relates set
  template <typename T>
  set<T> intersect(const set<T>& s, const set<T>& t) {
    return s.intersect(t);
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_SET_HPP
