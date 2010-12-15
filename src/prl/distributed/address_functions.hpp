#ifndef PRL_ADDRESS_FUNCTIONS_HPP
#define PRL_ADDRESS_FUNCTIONS_HPP

#include <string>

namespace prl {

  //! Returns the string representing an ip address of a host
  //! Throws std::runtime_error if the
  std::string address_string(const std::string& hostname);

}

#endif
