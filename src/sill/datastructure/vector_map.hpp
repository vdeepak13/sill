#ifndef SILL_VECTOR_MAP_HPP
#define SILL_VECTOR_MAP_HPP

#include <vector>
#include <utility>
#include <algorithm>

namespace sill {

  namespace detail {

    // less-than compares two entries in the vector map
    template <typename TLess>
    struct vector_map_less {
      vector_map_less(TLess tless) : tless(tless) { }
      template <typename K, typename T>
      bool operator()(const std::pair<K, T>& a, const std::pair<K, T>& b) {
        if (a.first == b.first) {
          return tless(a.second, b.second);
        } else {
          return a.first < b.first;
        }
      }
    private:
      TLess tless;
    };

    // equality compares two entries in the vector map
    template <typename TLess>
    struct vector_map_equal {
      vector_map_equal(TLess tless) : tless(tless) { }
      template <typename K, typename T>
      bool operator()(const std::pair<K, T>& a, const std::pair<K, T>& b) {
        return a.first == b.first 
          && !tless(a.second, b.second)
          && !tless(b.second, a.second);
      }
    private:
      TLess tless;
    };

  }

  //! A simple map based on std::vector
  //! \ingroup datastructure
  template <typename Key, typename T>
  class vector_map : public std::vector< std::pair<Key, T> >{
  public:
    typedef std::vector< std::pair<Key, T > > base;

    // public type declarations
    typedef Key key_type;
    typedef T   mapped_type;

    // shortcuts
    typedef typename base::value_type     value_type;
    typedef typename base::const_iterator const_iterator;
    typedef typename base::iterator       iterator;

    using base::begin;
    using base::end;

    const_iterator find(const Key& key) const {
      const_iterator it = begin(), end = this->end();
      for(; it != end; ++it) if (it->first == key) return it;
      return end;
    }

    iterator find(const Key& key) {
      iterator it = begin(), end = this->end();
      for(; it != end; ++it) if (it->first == key) return it;
      return end;
    }
      
    T& operator[](const Key& key) {
      iterator it = find(key);
      if (it == end()) { // expand the vector
        base::push_back(std::make_pair(key, T()));
        return base::back().second;
      }
      else return it->second;
    }

    size_t erase(const Key& key) {
      iterator it = find(key);
      if (it == end()) return 0;
      else {
        base::erase(it);
        return 1;
      }
    }

    void insert(const value_type& value) {
      base::push_back(value);
    }

    //! Sorts the elements in the map by key and then using tless
    //! invalidates the iterators
    template <typename TLess>
    void sort(TLess tless) const {
      vector_map& b = const_cast<vector_map&>(*this);
      std::sort(b.begin(), b.end(), detail::vector_map_less<TLess>(tless));
      // this should really use a lambda function
    }
    
  }; // class vector_map


  //! Returns true if the two maps are equal
  //! invalidates the iterators
  template <typename Key, typename T, typename TLess>
  bool equal(const vector_map<Key, T>& a,
             const vector_map<Key, T>& b,
             TLess tless) {
    if (a.size() != b.size()) {
      return false;
    }
    a.sort(tless);
    b.sort(tless);
    return std::equal(a.begin(), a.end(), b.begin(),
                      detail::vector_map_equal<TLess>(tless));
    // use a lambda function here, too
  }

} // namespace sill

#endif
