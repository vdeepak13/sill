#ifndef SILL_ADD_HPP
#define SILL_ADD_HPP

#include <sill/functional/aggregate_op.hpp>

namespace sill {
  
  template <typename T>
  struct add : public aggregate_op<T> {
    void operator()(const T& val, T& agg) const {
      agg += val;
    }
  };

}

#endif
