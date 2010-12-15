#include <stdexcept>
#include <arpa/inet.h>
#include <netdb.h>

#include <prl/distributed/address_functions.hpp>

namespace prl {

  std::string address_string(const std::string& hostname) {
    struct hostent* h = gethostbyname(hostname.c_str()); 
    if (!h) {
      throw std::runtime_error(std::string("Invalid address or hostname: ")
			       + hostname);
    }
    struct in_addr* ptr = (struct in_addr*)h->h_addr;
    return inet_ntoa(*ptr);
  }

}

