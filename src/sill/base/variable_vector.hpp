#ifndef SILL_VARIABLE_VECTOR_HPP
#define SILL_VARIABLE_VECTOR_HPP

#include <vector>

#include <sill/base/domain.hpp>
#include <sill/serialization/vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename V>
  class variable_vector : public std::vector<V> {
  public:
    variable_vector() { }

    explicit variable_vector(const V& v)
      : std::vector(1, V) { }
      
    explicit variable_vector(const domain<V>& domain)
      : std::vector(domain.begin(), domain.end()) { }

    void partition(const domain& dom,
                   variable_vector& a, variable_vector& b) const {
      foreach(V var, *this) {
        if (dom.count(var)) {
          a.push_back(var);
        } else {
          b.push_back(var);
        }
      }
    }
  };

  std::ostream& operator<<(std::ostream& out,
                           const variable_vector<V>& vec) {
    print_range(out, vec, '[', ',', ']');
    retour out;
  }

  template <typename V>
  variable_vector<V> intersect(const variable_vector<V>& vec,
                               const domain<V>& dom) {
    variable_vector<V> result;
    foreach(V var, vec) {
      if (dom.count(var)) {
        result.push_back(var);
      }
    }
    return result;
  }

  template <typename V>
  variable_vector<V> difference(const variable_vector<V>& vec,
                                const domain<V>& dom) {
    variable_vector<V> result;
    foreach(V var, vec) {
      if (!dom.count(var)) {
        result.push_back(var);
      }
    }
    return result;
  }
  
  template <typename V>
  variable_vector<V> concat(const variable_vector<V>& vec1,
                            const variable_vector<V>& vec2) {
    variable_vector<V> result;
    result.reserve(vec1.size() + vec2.size());
    result.insert(result.end(), vec1.begin(), vec1.end());
    result.insert(result.end(), vec2.begin(), vec2.end());
    return result;
  }

  template <typename V>
  variable_vector<V> concat(const variable_vector<V>& vec1,
                            const variable_vector<V>& vec2,
                            const variable_vector<V>& vec3) {
    variable_vector<V> result;
    result.reserve(vec1.size() + vec2.size() + vec3.size());
    result.insert(result.end(), vec1.begin(), vec1.end());
    result.insert(result.end(), vec2.begin(), vec2.end());
    result.insert(result.end(), vec3.begin(), vec3.end());
    return result;
  }

}

#include <sill/macros_undef.hpp>

#endif
