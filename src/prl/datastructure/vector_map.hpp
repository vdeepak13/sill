#ifndef PRL_VECTOR_MAP_HPP
#define PRL_VECTOR_MAP_HPP

#include <vector>
#include <utility>

namespace prl {

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

  }; // class vector_map

} // namespace prl

#endif
