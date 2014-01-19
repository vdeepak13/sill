#ifndef SILL_RECIPROCAL_HPP
#define SILL_RECIPROCAL_HPP

#include <sill/macros_def.hpp>

namespace sill {

  //! A functor which returns the reciprocal of the given number.
  template <typename T>
  struct reciprocal_functor {

    BOOST_STATIC_ASSERT(std::numeric_limits<T>::has_infinity);

    T operator()(const T& val) const {
      if (val != 0)
        return 1. / val;
      else
        return std::numeric_limits<T>::infinity();
    }

  };

}

#include <sill/macros_undef.hpp>

#endif // SILL_RECIPROCAL_HPP

