#ifndef SILL_XML_TAG_HPP
#define SILL_XML_TAG_HPP

#include <sill/named_value_pair.hpp>

namespace sill {

  // Type names for elementary datatypes
  //============================================================================
  inline const char* xml_tag(double*) {
    return "double";
  }
  
  inline const char* xml_tag(float*) {
    return "float";
  }

}

#endif
