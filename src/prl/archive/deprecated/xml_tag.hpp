#ifndef PRL_XML_TAG_HPP
#define PRL_XML_TAG_HPP

#include <prl/named_value_pair.hpp>

namespace prl {

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
