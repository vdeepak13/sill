#ifndef PRL_ADD_HPP
#define PRL_ADD_HPP

#include <prl/functional/aggregate_op.hpp>

namespace prl {
  
  template <typename T>
  struct add : public aggregate_op<T> {
    void operator()(const T& val, T& agg) const {
      agg += val;
    }
  };

}

#endif
