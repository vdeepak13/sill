#ifndef PRL_CONSTANT_FACTOR_XML_HPP
#define PRL_CONSTANT_FACTOR_XML_HPP

#include <iosfwd>

#include <prl/factor/constant_factor.hpp>
#include <prl/archive/xml_tag.hpp>
#include <prl/archive/xml_oarchive.hpp>
#include <prl/archive/xml_iarchive.hpp>

namespace prl {

  template <typename T>
  const char* xml_tag(constant_factor*) { 
    return "constant_factor";
  }

  template <typename T>
  xml_oarchive& operator<<(xml_oarchive& out, const constant_factor& f) {
    out.save_begin("constant_factor");
    out.add_attribute("storage", xml_tag((T*)0));
    out << f.value;
    out.save_end();
    return out;
  }

  template <typename T>
  xml_iarchive& operator>>(xml_iarchive& in, constant_factor& f) {
    in.load_begin("constant_factor");
    in >> f.value;
    in.load_end();
    return in;
  }

} // namespace prl

#endif
