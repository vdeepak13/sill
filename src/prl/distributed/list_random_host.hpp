#ifndef PRL_LIST_RANDOM_HOST_HPP
#define PRL_LIST_RANDOM_HOST_HPP

#include <vector>
#include <string>

#include <prl/distributed/random_host.hpp>

namespace prl {

  /**
   * A function object that returns a random host-port pairs from 
   * a pre-loaded list.
   *
   * Note that this class uses the standard rand() function,
   * so the user may need to set the seed using srand().
   */
  class list_random_host : public random_host {

    std::vector< std::pair<std::string, unsigned short> > hosts;
    
  public:
    //! Loads a list of host-port pairs from a file
    list_random_host(const char* filename);

    //! Loads a list of hosts from a file and adds a fixed port to each host
    list_random_host(const char* filename, unsigned short port);

    //! Returns a random host-port  
    std::pair<std::string, unsigned short> operator()() const;
    
  };

}

#endif
