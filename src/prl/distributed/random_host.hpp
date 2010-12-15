#ifndef PRL_RANDOM_HOST_HPP
#define PRL_RANDOM_HOST_HPP

namespace prl {

  //! An interface of a function that returns a random host-port pair
  struct random_host {
    virtual std::pair<std::string, unsigned short> operator()() const = 0;
    virtual ~random_host() { }
  };

}

#endif
