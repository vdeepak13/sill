#ifndef PRL_RECIPROCAL_HPP
#define PRL_RECIPROCAL_HPP

#include <prl/macros_def.hpp>

namespace prl {

  //! A functor which returns the reciprocal of the given number.
  template <typename T>
  struct reciprocal_functor {

    static_assert(std::numeric_limits<T>::has_infinity);

    T operator()(const T& val) const {
      if (val != 0)
        return 1. / val;
      else
        return std::numeric_limits<T>::infinity();
    }

  };

}

#include <prl/macros_undef.hpp>

#endif // PRL_RECIPROCAL_HPP

