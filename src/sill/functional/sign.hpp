#ifndef SILL_SIGN_HPP
#define SILL_SIGN_HPP

namespace sill {

  //! Functor which returns the sign of a value.
  //! Sign(x) = {-1 if x<0, 0 if x==0, 1 if x>0}
  template <typename T>
  struct sign {
    T operator()(const T& val) const {
      if (val < 0)
        return (T)(-1);
      else if (val == 0)
        return 0;
      else
        return 1;
    }
  };

}

#endif
