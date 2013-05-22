#ifndef SILL_SIZE_LESS_HPP
#define SILL_SIZE_LESS_HPP

#include <functional>

namespace sill {
  
  //! A functor that compares the sizes of two containers
  template <typename Container>
  struct size_less {

    //! Returns x.size() < y.size()
    bool operator()(const Container& x, const Container& y) const {
      return x.size() < y.size();
    }
  };

}

#endif
