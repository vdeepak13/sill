#include <iostream>

#include <boost/lexical_cast.hpp>

#include <sill/base/finite_variable.hpp>
#include <sill/base/finite_assignment_iterator.hpp>
#include <sill/base/universe.hpp>
#include <sill/math/operations.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  std::string finite_variable::str() const {
    return "#V(" + name() + "|F|" 
      + boost::lexical_cast<std::string>(size()) + ")";
  }

  bool finite_variable::type_compatible(finite_variable* v) const {
    assert(v);
    return size() == v->size();
  }
  
  bool finite_variable::type_compatible(variable* v) const {
    finite_variable* fv = dynamic_cast<finite_variable*>(v);
    return fv && size() == fv->size();
  }

  size_t finite_variable::value(const std::string& str, size_t offset) const {
    size_t val = boost::lexical_cast<size_t>(str);
    val -= offset;
    assert(val < size());
    return val;
  }
  
  void finite_variable::save(oarchive & ar) const {
    variable::save(ar);
    ar << size_;
  }
  
  void finite_variable::load(iarchive & ar) {
    variable::load(ar);
    ar >> size_;
  }

  size_t num_assignments(const finite_domain& vars) {
    size_t count = 1;
    foreach(finite_variable* v, vars) {
      if (std::numeric_limits<size_t>::max() / v->size() <= count) {
        throw std::out_of_range("num_assignments possibly overflows size_t");
      }
      count *= v->size();
    }
    return count;
  }

  size_t num_assignments(const finite_var_vector& vars) {
    size_t count = 1;
    foreach(finite_variable* v, vars) {
      if (std::numeric_limits<size_t>::max() / v->size() <= count) {
        throw std::out_of_range("num_assignments possibly overflows size_t");
      }
      count *= v->size();
    }
    return count;
  }

  oarchive& operator<<(oarchive& ar, finite_variable* const &v){
    // use the standard variable* serializer
    ar << (dynamic_cast<variable* const>(v));
    return ar;
  }
  

  iarchive& operator>>(iarchive& ar, finite_variable* &v){
    // use the standard variable* deserializer
    variable *tmp = NULL;
    ar >> tmp;
    v = dynamic_cast<finite_variable* >(tmp);
    return ar;
  }

} // namespace sill

