#ifndef SILL_AGGREGATE_OP_HPP
#define SILL_AGGREGATE_OP_HPP

namespace sill {

  template <typename T>
  struct aggregate_op {
    virtual void operator()(const T& value, T& aggregate) const = 0;
  };

}

#endif
