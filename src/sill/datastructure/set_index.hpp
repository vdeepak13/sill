
#ifndef SILL_SET_INDEX_HPP
#define SILL_SET_INDEX_HPP

#include <list>
#include <map>
#include <set>
#include <stdexcept>

#include <sill/global.hpp>
#include <sill/stl_concepts.hpp>
#include <sill/iterator/counting_output_iterator.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/serialization/list.hpp>
#include <sill/serialization/map.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * An index over sets that efficiently processes intersection and
   * superset queries and which supports efficient insertion and
   * deletion of sets.  Each set can be associated with a handle
   * that is returned during the superset and intersection queries.
   *
   * @param Set
   *        The type of stored in the index.
   *        TODO: fix the concepts or parameterize by the element type.
   * @param Handle
   *        Handle associated with each set in the index.
   *        Must be DefaultConstructible, CopyConstructible, and Assignable.
   *
   * \ingroup datastructure
   */
  template <typename Set, typename Handle = void_>
  class set_index {
    concept_assert((DefaultConstructible<Handle>));
    concept_assert((CopyConstructible<Handle>));
    concept_assert((Assignable<Handle>));

    // Private type declarations and members
    //==========================================================================
  private:
    //! The type of elements stored in each set
    typedef typename Set::value_type element_type;

    //! The set paired eith the handle
    typedef std::pair<Set, Handle> set_handle_pair;

    //! The type that maps each element to the sets that contain that element.
    typedef std::map<element_type, std::list<set_handle_pair> > element_set_map;

    //! The primary index structure. 
    //! The lists pointed to by this structure are guaranteed to be not empty.
    element_set_map element_sets;

    //! The list of indexed sets that are empty.
    std::list<set_handle_pair> empty_sets;


    // Public type declarations (copy the template argument types)
    //==========================================================================
  public:
    //! The set type
    typedef Set set_type;

    //! The handle type
    typedef Handle handle_type;


  private:
    template<typename T>
    static bool is_lce(const T& e, const std::set<T>& r, const std::set<T>& s) {
      typename std::set<T>::const_iterator r_it = r.begin(), 
                                          r_end = r.end(), 
                                          s_it = s.begin(), 
                                          s_end = s.end();
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

    // Constructors and destructors
    //==========================================================================
  public:
    //! Default constructor; the index has no sets in it.
    set_index() { }

    //! Swaps the content of two index sets (in constant time).
    void swap(set_index& other) {
      element_sets.swap(other.element_sets);
      empty_sets.swap(other.empty_sets);
    }

    //! Serialize members
    void save(oarchive & ar) const {
      ar << element_sets << empty_sets;
    }

    //! Deserialize members
    void load(iarchive & ar) {
      ar >> element_sets >> empty_sets;
    }

    // Queries
    //==========================================================================
    
    //! Returns true if the index contains no sets
    bool empty() const {
      return empty_sets.empty() && element_sets.empty();
    }

    //! Returns the handle of the first stored element
    const Handle& front() const {
      if (!empty_sets.empty())
        return empty_sets.front().second;
      else if (!element_sets.empty())
        return element_sets.begin()->second.front().second;
      else
        assert(false);
    }

    /**
     * Returns a handle for any set that contains the specified element.
     * \throw std::invalid_argument if no there is no set containing the element
     */
    const Handle& operator[](element_type elt) const {
      return safe_get(element_sets, elt).begin()->second;
    }

    /**
     * Returns the number of sets that contain an element.
     */
    size_t count(element_type elt) const {
      if (element_sets.count(elt))
        return safe_get(element_sets, elt).size();
      else
        return 0;
    }
    
    /**
     * Intersection query.  The handle for each set in this index that
     * intersects the supplied set is written to the output iterator.
     *
     * @param set the set whose intersecting sets are desired
     * @param output the output iterator to which handles for all
     *               intersecting sets in this index are written
     */
    template <typename OutIt>
    OutIt find_intersecting_sets(const Set& set, OutIt out) const {
      concept_assert((boost::OutputIterator<OutIt, Handle>));
      foreach(element_type elt, set) {
        if (element_sets.count(elt)) {
          foreach(const set_handle_pair& set_handle, safe_get(element_sets,elt)) {
            // To avoid writing sets multiple times, check to see if
            // the least common element of the two sets is elt.
            if (is_lce(elt, set, set_handle.first)) {
              *out = set_handle.second;
              ++out;
            }
          }
        }
      }
      return out;
    }

    /**
     * Superset query.  The handle for each set in this index that
     * contains the supplied set is written to the output iterator.
     *
     * @param set the set whose supersets are desired
     * @param output the output iterator to which handles for all
     *               supersets in this index are written
     */
    template <typename OutIt>
    OutIt find_supersets(const Set& set, OutIt out) const {
      concept_assert((boost::OutputIterator<OutIt, Handle>));

      if (set.empty()) { 
        // Every set in the index is a superset of an empty set
        foreach(const set_handle_pair& set_handle, empty_sets) {
          *out = set_handle.second;
          ++out;
        }
        foreach(typename element_set_map::const_reference ref, element_sets) {
          foreach(const set_handle_pair& set_handle, ref.second) {
            // make sure that each set is included only once
            if(*(set_handle.first.begin()) == ref.first) {
              *out = set_handle.second;
              ++out;
            }
          }
        }
      } else {
        // Pick one element and iterate over all sets that contain that element
        element_type elt = *(set.begin());
        if (element_sets.count(elt)) {
          foreach(const set_handle_pair& set_handle, safe_get(element_sets, elt)) {
            if(includes(set_handle.first, set)) {
              *out = set_handle.second;
              ++out;
            }
          }
        }
      }

      return out;
    }

    /**
     * Minimal superset query: find a minimal set which is a superset of the
     * supplied set. If the set supplied set is empty, this operation 
     * may return any set.
     * 
     * @param  set the set whose supersets are desired
     * @return the handle for a minimal superset if a superset exists.
     *         Returns Handle() if no superset exists.
     */
    Handle find_min_cover(const Set& set) const {
      if (set.empty()) {
        if (!empty_sets.empty()) 
          return empty_sets.begin()->second;
        if (!element_sets.empty())
          return element_sets.begin()->second.begin()->second;
        return Handle();
      }

      element_type elt = *(set.begin());
      size_t min_size = std::numeric_limits<size_t>::max();
      Handle result   = Handle();
      if (element_sets.count(elt)) {
        foreach(const set_handle_pair& set_handle, safe_get(element_sets, elt)) {
          if (set_handle.first.size() < min_size
              && includes(set_handle.first, set)) {
            min_size = set_handle.first.size();
            result   = set_handle.second;
          }
        }
      }

      return result;
    }

    /**
     * Maximal intersection query: find a set whose intersection with
     * the supplied set is maximal and non-zero.
     * 
     * @param set the set intersection with which are desired
     * @return the handle for a set with a non-zero maximal intersection
     *         with the supplied set. Of the sets with the same 
     *         intersection size, returns the smallest one.
     *         Otherwise, returns Handle().
     */
    Handle find_max_intersection(const Set& set) const {
      if (set.empty()) return Handle();
      
      Handle result = Handle();
      size_t max_intersection = 0;
      size_t min_size = std::numeric_limits<size_t>::max();
      foreach(element_type elt, set) {
        if (element_sets.count(elt)) {
           foreach(const set_handle_pair& set_handle, safe_get(element_sets, elt)) {
             // To avoid checking sets multiple times, check to see if
             // the least common element of the two sets is elt.
             if (is_lce(elt, set, set_handle.first)) {
               size_t intersection = set_intersect(set, set_handle.first).size();
               if ((intersection > max_intersection) ||
                   (intersection == max_intersection && 
                    set_handle.first.size() < min_size)) {
                 max_intersection = intersection;
                 min_size = set_handle.first.size();
                 result   = set_handle.second;
               }
             }
           }
        }
      }

      return result;
    }

    /**
     * Returns true if there are no supersets of the supplied set in
     * this index.
     *
     * @param  set the set whose maximality is tested
     * @return true iff the set is maximal w.r.t. the sets in this
     *         index
     */
    bool is_maximal(const Set& set) const {
      counting_output_iterator it;
      return find_supersets(set, it).count() == 0;
    }

    //! Prints *this to an output stream
    template <typename OutputStream>
    void print(OutputStream& out) const {
      out << "[";
      foreach(typename element_set_map::const_reference ref, element_sets)
        out << ref.first << "-->" << ref.second << std::endl;
      out << empty_sets << "]" << std::endl;
    }

    // Mutating operations
    //==========================================================================

    /**
     * Inserts a new set in the index.  This function runs in
     * \f$O(k)\f$ time, where \f$k\f$ is the number of elements in the
     * supplied set.
     *
     * @param  set the set to be inserted
     * @param  handle the handle associated with the set
     */
    void insert(const Set& set, const Handle& handle = Handle()) {
      set_handle_pair set_handle(set, handle);
      if (set.empty())
        empty_sets.push_front(set_handle);
      else {
        foreach(element_type elt, set)
          element_sets[elt].push_front(set_handle);
      }
    }

    //! Removes all pairs (set, handle) from the index.
    void remove(const Set& set, const Handle& handle = Handle()) {
      set_handle_pair set_handle(set, handle);
      if (set.empty())
        empty_sets.remove(set_handle);
      else {
        foreach(element_type elt, set) {
          typename element_set_map::iterator it = element_sets.find(elt);
          assert(it != element_sets.end());
          it->second.remove(set_handle);
          if (it->second.empty())
            element_sets.erase(it);
        }
      }
    }
    
    //! Removes all sets from this index
    void clear() {
      element_sets.clear();
      empty_sets.clear();
    }

  }; // class set_index

  template <typename Set, typename Handle>
  std::ostream& operator<<(std::ostream& out, 
                           const set_index<Set, Handle>& index) {
    index.print(out); 
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SET_INDEX_HPP
