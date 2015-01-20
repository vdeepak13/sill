#include <iostream>

#include <boost/lexical_cast.hpp>

#include <sill/base/universe.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/serialization/serialize.hpp>

namespace sill{

  // Vector variable
  //============================================================================
  std::string vector_variable::str() const {
    return "#V(" + name() + "|V|"
      + boost::lexical_cast<std::string>(size()) + ")";
  }
  
  bool vector_variable::type_compatible(vector_variable* x) const {
    assert(x);
    return size() == x->size();
  }

  bool vector_variable::type_compatible(variable* v) const {
    vector_variable* vv = dynamic_cast<vector_variable*>(v);
    return vv && size() == vv->size();
  }
  
  void vector_variable::save(oarchive& ar) const {
    variable::save(ar);
    ar << size_;
  }
  void vector_variable::load(iarchive& ar) {
    variable::load(ar);
    ar >> size_;
  }
  
  oarchive& operator<<(oarchive& ar, vector_variable* const& v) {
    // use the standard variable* serializer
    ar << (dynamic_cast<variable* const>(v));
    return ar;
  }
  
  iarchive& operator>>(iarchive& ar, vector_variable* &v) {
    // use the standard variable* deserializer
    variable* tmp = NULL;
    ar >> tmp;
    v = dynamic_cast<vector_variable*>(tmp);
    return ar;
  }
  
  /*
  vec vector_variable::value(const std::string& str) const {
    //value_type vec = boost::lexical_cast<value_type>(str);
    //assert(vec.size() == size());
    //return vec;
    assert(false);
    return value_type();
  }
  */
}
