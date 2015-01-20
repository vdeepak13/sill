#ifndef SILL_STL_UTIL_HPP
#define SILL_STL_UTIL_HPP

#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <sill/iterator/counting_output_iterator.hpp>
#include <sill/iterator/map_value_iterator.hpp>
#include <sill/serialization/map.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/set.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

/**
 * \file stl_util.hpp  Utilities for STL classes
 *
 * File contents:
 *  - Functions on sets
 *  - Functions on maps
 *  - Functions on vectors
 */

namespace sill {

  // Functions on sets
  //============================================================================

  /**
   * computes the union of two sets.
   */
  template <typename T>
  std::set<T> set_union(const std::set<T>& a, const std::set<T>& b) {
    std::set<T> output;
    std::set_union(a.begin(), a.end(), 
                   b.begin(), b.end(),
                   std::inserter(output, output.begin()));
    return output;
  }
  
  template <typename T>
  std::set<T> set_union(const std::set<T>& a, const T& b) {
    std::set<T> output = a;
    output.insert(b);
    return output;
  }

  template <typename T>
  std::set<T> set_intersect(const std::set<T>& a, const std::set<T>& b) {
    std::set<T> output;
    std::set_intersection(a.begin(), a.end(), 
                          b.begin(), b.end(),
                          std::inserter(output, output.begin()));
    return output;
  }

  template <typename T>
  size_t intersection_size(const std::set<T>& a, const std::set<T>& b) {
    counting_output_iterator counter;
    return std::set_intersection(a.begin(), a.end(), 
                                 b.begin(), b.end(),
                                 counter).count();
  }

  template <typename T>
  std::set<T> set_difference(const std::set<T>& a, const std::set<T>& b) {
    std::set<T> output;
    std::set_difference(a.begin(), a.end(), 
                        b.begin(), b.end(),
                        std::inserter(output, output.begin()));
    return output;
  }


  template <typename T>
  std::set<T> set_difference(const std::set<T>& a, const T& b) {
    std::set<T> output = a;
    output.erase(b);
    return output;
  }

  //! @return 2 sets: <s in partition, s not in partition>
  template <typename T>
  std::pair<std::set<T>,std::set<T> > 
  set_partition(const std::set<T>& s, const std::set<T>& partition) {
    std::set<T> a, b;
    a = set_intersect(s, partition);
    b = set_difference(s, partition);
    return std::make_pair(a, b);
  }

  template <typename T>
  bool set_disjoint(const std::set<T>& a, const std::set<T>& b) {
    return (intersection_size(a,b) == 0);
  }

  template <typename T>
  bool set_equal(const std::set<T>& a, const std::set<T>& b) {
    if (a.size() != b.size()) return false;
    return a == b; // defined in <set>
  }

  template <typename T>
  bool includes(const std::set<T>& a, const std::set<T>& b) {
    return std::includes(a.begin(), a.end(), b.begin(), b.end());
  }

  template <typename T>
  bool is_subset(const std::set<T>& a, const std::set<T>& b) {
    return includes(b, a);
  }

  template <typename T>
  bool is_superset(const std::set<T>& a,const std::set<T>& b) {
    return includes(a, b);
  }

  //! Writes a human representation of the set to the supplied stream.
  //! \relates set
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const std::set<T>& s) {
    return print_range(out, s, '{', ' ', '}');
  }

  // Functions on maps
  //============================================================================

  /**
   * constant lookup in a map. assertion failure of key not found in map
   */
  template <typename Key, typename T>
  const T& safe_get(const std::map<Key, T>& map,
                    const Key& key) {
    typedef typename std::map<Key, T>::const_iterator iterator;
    iterator iter = map.find(key);
    assert(iter != map.end());
    return iter->second;
  } // end of safe_get

  /**
   * constant lookup in a map. If key is not found in map, 
   * 'default_value' is returned. Note that this can't return a reference
   * and must return a copy
   */
  template <typename Key, typename T>
  const T safe_get(const std::map<Key, T>& map,
                   const Key& key, const T default_value) {
    typedef typename std::map<Key, T>::const_iterator iterator;
    iterator iter = map.find(key);
    if (iter == map.end()) {
      return default_value;
    }
    else {
      return iter->second;
    }
  } // end of safe_get

  /**
   * Returns a pointer to an element in a map if the value exists
   * and NULL otherwise.
   */
  template <typename Map>
  const typename Map::mapped_type*
  get_ptr(const Map& map, const typename Map::key_type& key) {
    typename Map::const_iterator it = map.find(key);
    if (it == map.end()) {
      return NULL;
    } else {
      return &it->second;
    }
  }

  /**
   * Transform each key in the map using the key_map
   * transformation. The resulting map will have the form
   * output[key_map[i]] = map[i]
   */
  template <typename OldKey, typename NewKey, typename T>
  std::map<NewKey, T>
  rekey(const std::map<OldKey, T>& map,
        const std::map<OldKey, NewKey>& key_map) {
    std::map<NewKey, T> output;
    typedef std::pair<OldKey, T> pair_type;
    foreach(const pair_type& pair, map) {
      output[safe_get(key_map, pair.first)] = pair.second;
    }
    return output;
  }

  /**
   * Transform each key in the map using the key_map
   * transformation. The resulting map will have the form
   * output[i] = remap[map[i]]
   */
  template <typename Key, typename OldT, typename NewT>
  std::map<Key, NewT>
  remap(const std::map<Key, OldT>& map,
        const std::map<OldT, NewT>& val_map) {
    std::map<Key, NewT> output;
    typedef std::pair<Key, OldT> pair_type;
    foreach(const pair_type& pair, map) {
      output[pair.first] = safe_get(val_map, pair.second);
    }
    return output;
  }

// it's impossible to differentiate this from the above function
//   /**
//    * Inplace version of remap
//    */
//   template <typename Key, typename T>
//   void remap(std::map<Key, T>& map,
//              const std::map<T, T>& val_map) {
//     typedef std::pair<Key, T> pair_type;
//     foreach(pair_type& pair, map) {
//       pair.second = safe_get(val_map, pair.second);
//     }
//   }

  /**
   * Inserts the elements in map 'from' into map 'to',
   * overwriting any existing elements with the same keys.
   * Note: std::map::insert does not behave this way for some reason.
   */
  template <typename Key, typename T>
  void map_insert(const std::map<Key, T>& from,
                  std::map<Key, T>& to) {
    typename std::map<Key,T>::const_iterator from_it(from.begin());
    typename std::map<Key,T>::const_iterator from_end(from.end());
    while (from_it != from_end) {
      to[from_it->first] = from_it->second;
      ++from_it;
    }
  }

  /**
   * Computes the union of two maps
   */
  template <typename Key, typename T>
  std::map<Key, T> 
  map_union(const std::map<Key, T>& a,
            const std::map<Key, T>& b) {
    // Initialize the output map
    std::map<Key, T> output;
    std::set_union(a.begin(), a.end(),
                   b.begin(), b.end(),
                   std::inserter(output, output.begin()),
                   output.value_comp());
    return output;
  }
  
  /**
   * Computes the intersection of two maps
   */
  template <typename Key, typename T>
  std::map<Key, T> 
  map_intersect(const std::map<Key, T>& a,
                const std::map<Key, T>& b) {
    // Initialize the output map
    std::map<Key, T> output;
    // compute the intersection
    std::set_intersection(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::inserter(output, output.begin()),
                          output.value_comp());
    return output;
  }
  
  /**
   * Returns the entries of a map whose keys show up in the set keys
   */
  template <typename Key, typename T>
  std::map<Key, T> 
  map_intersect(const std::map<Key, T>& m,
                const std::set<Key>& keys) {
    std::map<Key, T> output;
    foreach(const Key& key, keys) {
      typename std::map<Key,T>::const_iterator it = m.find(key);
      if (it != m.end())
        output[key] = it->second;
    }
    return output;
  }

  /**
   * Computes the difference between two maps
   */
  template <typename Key, typename T>
  std::map<Key, T> 
  map_difference(const std::map<Key, T>& a,
                 const std::map<Key, T>& b) {
    // Initialize the output map
    std::map<Key, T> output;
    // compute the intersection
    std::set_difference(a.begin(), a.end(),
                        b.begin(), b.end(),
                        std::inserter(output, output.begin()),
                        output.value_comp());
    return output;
  }


  /**
   * Returns the set of keys in a map
   */
  template <typename Key, typename T>
  std::set<Key> keys(const std::map<Key, T>& map) {
    std::set<Key> output;
    typedef std::pair<Key, T> pair_type;
    foreach(const pair_type& pair, map) {
      output.insert(pair.first);
    }
    return output;
  }

  /**
   * Gets the values from a map
   */
  template <typename Key, typename T>
  std::pair<map_value_iterator<std::map<Key,T> >,
            map_value_iterator<std::map<Key,T> > >
  values(const std::map<Key, T>& map) {
    return std::make_pair(map_value_iterator<std::map<Key,T> >(map.begin()),
                          map_value_iterator<std::map<Key,T> >(map.end()));
  }

  template <typename Key, typename T>
  std::vector<T> values(const std::map<Key, T>& m, 
                        const std::set<Key>& keys) {
    std::vector<T> output;

    foreach(const Key &i, keys) {
      output.push_back(safe_get(m, i));
    }
    return output;
  }
  
  template <typename Key, typename T>
  std::vector<T> values(const std::map<Key, T>& m, 
                        const std::vector<Key>& keys) {
    std::vector<T> output;
    foreach(const Key &i, keys) {
      output.push_back(safe_get(m, i));
    }
    return output;
  }
  
  //! Creates an identity map (a map from elements to themselves)
  //! \relates map
  template <typename Key>
  std::map<Key, Key> make_identity_map(const std::set<Key>& keys) {
    std::map<Key, Key> m;
    foreach(const Key& key, keys) 
      m[key] = key;
    return m;
  }

  //! Writes a map to the supplied stream.
  //! \relates map
  template <typename Key, typename T>
  std::ostream& operator<<(std::ostream& out, const std::map<Key, T>& m) {
    out << "{";
    for (typename std::map<Key, T>::const_iterator it = m.begin(); 
         it != m.end();) {
      out << it->first << "-->" << it->second;
      if (++it != m.end()) out << " ";
    }
    out << "}";
    return out;
  }

  // Functions on vectors
  //============================================================================

  //! Concatenates two std::vectors.
  //! \relates std::vector
  template <typename T>
  std::vector<T> concat(const std::vector<T>& vec1,
                        const std::vector<T>& vec2) {
    std::vector<T> vec;
    vec.reserve(vec1.size() + vec2.size());
    vec.insert(vec.end(), vec1.begin(), vec1.end());
    vec.insert(vec.end(), vec2.begin(), vec2.end());
    return vec;
  }

  //! Concatenates three std::vectors.
  //! \relates std::vector
  template <typename T>
  std::vector<T> concat(const std::vector<T>& vec1,
                        const std::vector<T>& vec2,
                        const std::vector<T>& vec3) {
    std::vector<T> vec;
    vec.reserve(vec1.size() + vec2.size() + vec3.size());
    vec.insert(vec.end(), vec1.begin(), vec1.end());
    vec.insert(vec.end(), vec2.begin(), vec2.end());
    vec.insert(vec.end(), vec3.begin(), vec3.end());
    return vec;
  }

  /**
   * Returns true if vector x is a prefix of vector y.
   */
  template <typename T>
  bool prefix(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() <= y.size() && std::equal(x.begin(), x.end(), y.begin());
  }

  /**
   * Returns the union of two vectors.
   */
  template <typename T>
  std::vector<T> set_union(const std::vector<T>& x,
                           const std::vector<T>& y) {
    std::vector<T> result = x;
    foreach (T v, y) {
      if (!std::count(x.begin(), x.end(), v)) {
        result.push_back(v);
      }
    }
    return result;
  }

  /**
   * Lexigraphical comparison of two ranges.
   * Reading left to right, upon the first element j for which a,b differ,
   * this function returns -1 if a[j] < b[j] and +1 if a[j] > b[j].
   * If a is a prefix of b, this returns -1.
   * If b is a prefix of a, this returns +1.
   * If a,b are exactly the same, this returns 0.
   * @todo Test this!
   */
  /*
  template <typename T>
  int lexigraphic_compare(const forward_range<T>& a,
                          const forward_range<T>& b) {
    const T* a_it = a.begin();
    const T* a_end = a.end();
    const T* b_it = b.begin();
    const T* b_end = b.end();
    while (a_it != a_end && b_it != b_end) {
      if (*a_it < *b_it)
        return -1;
      if (*a_it > *b_it)
        return 1;
      ++a_it;
      ++b_it;
    }
    if (b_it != b_end)
      return -1;
    if (a_it != a_end)
      return 1;
    return 0;
  }
  */

  //! Returns a subvector of v specified by the complement of the given indices.
  //! @param indices_sorted  Specifies if the given indices are sorted in
  //!                        in increasing order.
  template <typename T, typename IndexType>
  std::vector<T> remove_subvector(const std::vector<T>& v,
                                  const std::vector<IndexType>& indices,
                                  bool indices_sorted = false) {
    std::vector<T> newv;
    if (indices_sorted) {
      size_t indices_i = 0;
      for (size_t i = 0; i < v.size(); ++i) {
        if (i < indices.size() &&
            i == indices[indices_i]) {
          ++indices_i;
        } else {
          newv.push_back(v[i]);
        }
      }
    } else {
      throw std::runtime_error("remove_subvector not yet fully implemented!");
    }
  } // remove_subvector

  //! Returns the subvector of v which also appears in set s.
  template <typename T>
  std::vector<T> select_subvector(const std::vector<T>& v,
                                  const std::set<T>& s) {
    std::vector<T> newv;
    foreach(const T& elem, v) {
      if (s.count(elem))
        newv.push_back(elem);
    }
    return newv;
  }

  //! Returns the subvector of v of elements which do not appear in set s.
  template <typename T>
  std::vector<T> select_subvector_complement(const std::vector<T>& v,
                                             const std::set<T>& s) {
    std::vector<T> newv;
    foreach(const T& elem, v) {
      if (s.count(elem) == 0)
        newv.push_back(elem);
    }
    return newv;
  }

  //! Returns the subvector of v which also appears in set s,
  //! as well as its complement.
  template <typename T>
  void select_subvector_and_complement(const std::vector<T>& v,
                                       const std::set<T>& s,
                                       std::vector<T>& subvec,
                                       std::vector<T>& subvec_complement) {
    subvec.clear();
    subvec_complement.clear();
    foreach(const T& elem, v) {
      if (s.count(elem))
        subvec.push_back(elem);
      else
        subvec_complement.push_back(elem);
    }
  }

  //! Builds an index from a vector of values, i.e.,
  //!  map[value] = index in vector
  //! NOTE: This requires all elements to be distinct!
  template <typename T>
  std::map<T,size_t> build_vector_index(const std::vector<T>& v) {
    std::map<T,size_t> idx;
    for (size_t i = 0; i < v.size(); ++i) {
      std::pair<typename std::map<T,size_t>::iterator, bool>
        it_inserted(idx.insert(std::make_pair(v[i], i)));
      if (!it_inserted.second) {
        throw std::runtime_error
          ("build_vector_index(v) was given v with duplicate elements!");
      }
    }
    return idx;
  }

} // end of namespace sill

#include <sill/macros_undef.hpp>

#endif
