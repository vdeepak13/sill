#ifndef SILL_POLYMORPHIC_FACTOR_XML_HPP
#define SILL_POLYMORPHIC_FACTOR_XML_HPP

#include <sill/factor/any_factor.hpp>
#include <sill/archive/xml_iarchive.hpp>
#include <sill/archive/xml_oarchive.hpp>

namespace sill {

  template <typename T>
  const char* xml_tag(any_factor<T>*) {
    return "any_factor";
  }

  template <typename T>
  xml_oarchive& operator<<(xml_oarchive& out, const any_factor<T>& f) {
    f.save(out);
    return out;
  }
  
  template <typename T>
  xml_iarchive& operator>>(xml_iarchive& in, any_factor<T>& f) {
    serializable* obj_ptr = in.read();
    f = dynamic_cast<factor&>(*obj_ptr);
    delete obj_ptr;
    return in;
  }

} // namespace sill 

#endif

