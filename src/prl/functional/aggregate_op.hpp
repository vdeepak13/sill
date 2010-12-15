#ifndef PRL_AGGREGATE_OP_HPP
#define PRL_AGGREGATE_OP_HPP

namespace prl {

  template <typename T>
  struct aggregate_op {
    virtual void operator()(const T& value, T& aggregate) const = 0;
  };

}

#endif
