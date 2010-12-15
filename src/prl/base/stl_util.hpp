
#ifndef PRL_STL_UTIL_HPP
#define PRL_STL_UTIL_HPP


#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sstream>

#include <prl/stl_io.hpp>
#include <prl/serialization/serialize.hpp>
#include <prl/serialization/set.hpp>
#include <prl/serialization/map.hpp>
#include <prl/iterator/counting_output_iterator.hpp>

#include <prl/macros_def.hpp>

namespace prl {

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
   output[i] = remap[map[i]]
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

  /**
   * Inplace version of remap
   */
  template <typename Key, typename T>
  void remap(std::map<Key, T>& map,
             const std::map<T, T>& val_map) {
    typedef std::pair<Key, T> pair_type;
    foreach(pair_type& pair, map) {
      pair.second = safe_get(val_map, pair.second);
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
  std::set<T> values(const std::map<Key, T>& map) {
    std::set<T> output;
    typedef std::pair<Key, T> pair_type;
    foreach(const pair_type& pair, map) {
      output.insert(pair.second);
    }
    return output;
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

}; // end of namespace prl

#include <prl/macros_undef.hpp>

#endif
