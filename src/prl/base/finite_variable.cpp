#include <iostream>

#include <boost/lexical_cast.hpp>

#include <prl/base/finite_variable.hpp>
#include <prl/base/finite_assignment_iterator.hpp>
#include <prl/base/universe.hpp>
#include <prl/math/free_functions.hpp>
#include <prl/macros_def.hpp>

namespace prl {

  finite_variable::operator std::string() const {
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
    double logn(log_num_assignments(vars));
    if (std::log(std::numeric_limits<size_t>::max()) < logn)
      throw std::out_of_range("num_assignments overflowed size_t");
    return (size_t)(round(std::exp(logn)));
  }

  double log_num_assignments(const finite_domain& vars) {
    double logn = 0.;
    foreach(finite_variable* v, vars)
      logn += std::log(v->size());
    return logn;
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

} // namespace prl

