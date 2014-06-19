#ifndef SILL_ASSIGNMENT_HPP
#define SILL_ASSIGNMENT_HPP

#include <map>

#include <boost/range/algorithm/set_algorithm.hpp>

#include <sill/base/domain.hpp>
#include <sill/base/variable_vector.hpp>
#include <sill/base/variable_map.hpp>
#include <sill/iterator/counting_output_iterator.hpp>
#include <sill/serialization/map.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // eventually, this template will just take the variable type
  template <typename V, typename T>
  class assignment : public std::map<V, T> {
  public:
    assignment() { }

    assignment(const V& var, const T& val) {
      this->insert(std::make_pair(var, val));
    }

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

    assignment rekey(const variable_map<V, V>& map) const {
      assignment result;
      foreach(typename std::map::const_reference pair, *this) {
        result[map.get(pair.first)] = pair.second;
      }
      return result;
    }
  };

  std::ostream& operator<<(std::ostream& out, const assignment<V,T>& a) {
    print_map(out, a, '{', ':', ',', '}');
    return out;
  }

  template <typename V, typename T>
  assignment<V,T>
  map_union(const assignment<V,T>& a, const assignment<V,T>& b) {
    assignment<V,T> result;
    boost::set_union(a, b, std::inserter(result, result.begin()),
                     result.value_comp());
    return result;
  }

  template <typename V, typename T>
  assignment<V,T>
  intersect(const assignment<V,T>& a, const assignment<V,T>& b) {
    assignment<V,T> result;
    boost::set_intersection(a, b, std::inserter(result, result.begin()),
                            result.value_comp());
    return result;
  }

  template <typename V, typename T>
  assignment<V,T>
  intersect(const assignment<V,T>& a, const domain<V,T>& dom) {
    assignment<V,T> result;
    foreach(V var, dom) {
      typename assignment<V,T>::const_iterator it = a.find(key);
      if (it != a.end()) {
        result.insert(*it);
      }
    }
    return result;
  }
  
  template <typename V, typename T>
  assignment<V,T>
  difference(const assignment<V,T>& a, const assignment<V,T>& b) {
    assignment<V,T> result;
    boost::set_difference(a, b, std::inserter(result, result.begin()),
                          result.value_comp());
    return result;
  }
  
  template <typename V, typename T>
  bool disjoint(const assignment<V,T>& a, const assignment<V,T>& b) {
    counting_output_iterator counter;
    return boost::set_intersection(a, b, counter,
                                   a.value_comp()).count() == 0;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
