#ifndef PRL_BOOST_UNORDERED_UTILS_HPP
#define PRL_BOOST_UNORDERED_UTILS_HPP

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  template <typename T>
  size_t intersection_size(const boost::unordered_set<T>& a,
                           const boost::unordered_set<T>& b) {
    size_t count = 0;
    if (a.size() < b.size()) {
      foreach(const T& key, a)
        count += b.count(key);
    } else {
      foreach(const T& key, b) 
        count += a.count(key);
    }
    return count;
  }

  template <typename Key, typename T>
  boost::unordered_set<Key> keys(const boost::unordered_map<Key, T>& map) {
    boost::unordered_set<Key> set;
    typedef std::pair<const Key, T> value_type;
    foreach(const value_type& val, map) 
      set.insert(val.first);
    return set;
  }
  
  template <typename Key, typename T>
  const T& get(const boost::unordered_map<Key, T>& map, const Key& key) {
    typename boost::unordered_map<Key, T>::const_iterator it = map.find(key);
    assert(it != map.end());
    return it->second;
  }

  template <typename Key, typename T>
  T& get(boost::unordered_map<Key, T>& map, const Key& key) {
    typename boost::unordered_map<Key, T>::iterator it = map.find(key);
    assert(it != map.end());
    return it->second;
  }
}

#include <prl/macros_undef.hpp>

#endif
