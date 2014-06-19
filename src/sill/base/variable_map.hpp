#ifndef SILL_VARIABLE_MAP_HPP
#define SILL_VARIABLE_MAP_HPP

#include <map>

#include <sill/base/domain.hpp>
#include <sill/base/variable_vector.hpp>
#include <sill/serialization/map.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename V, typename T>
  class variable_map : public std::map<V, T> {
  public:
    variable_map() { }

    const T& get(const V& var) const {
      typename std::map::const_iterator it = this->find(key);
      assert(it != this->end());
      return it->second;
    }

    const T get(const V& var, const T& default_value) const {
      typename std::map::const_iterator it = this->find(key);
      if (it == this->end()) {
        return default_value;
      } else {
        return it->second;
      }
    }

    domain<V> keys() const {
      domain<V> result;
      foreach(typename std::map::const_reference pair, *this) {
        result.insert(pair.first);
      }
      return result;
    }

    std::vector<T> values() const {
      std::vector<T> result;
      result.reserve(this->size());
      foreach(typename std::map::const_reference pair, *this) {
        result.push_back(pair.second);
      }
      return result;
    }

    std::vector<T> values(const domain<V>& vars) const {
      std::vector<T> result;
      result.reserve(this->size());
      foreach(V var, vars) {
        result.push_back(operator[](var));
      }
      return result;
    }

    std::vector<T> values(const variable_vector<V>& vars) const {
      std::vector<T> result;
      foreach(V var, vars) {
        result.push_back(operator[](var));
      }
      return result;
    }

    variable_map rekey(const variable_map<V, V>& map) const {
      variable_map result;
      foreach(typename std::map::const_reference pair, *this) {
        result[map.get(pair.first)] = pair.second;
      }
      return result;
    }
  };

  template <typename V, typename T>
  std::ostream& operator<<(std::ostream& out, const variable_map<V,T>& map) {
    print_map(out, a, '{', ':', ',', '}');
    return out;
  }

  template <typename V>
  variable_map<V,V> identity_map(const domain<V>& dom) {
    variable_map<V,V> result;
    foreach(V var, dom) {
      result.insert(std::make_pair(var, var));
    }
    return result;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
