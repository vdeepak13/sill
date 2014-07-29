#ifndef SILL_FACTOR_ESTIMATOR_HPP
#define SILL_FACTOR_ESTIMATOR_HPP

namespace sill {

  template <typename F>
  class factor_estimator {
  public:
    typedef typename F::domain_type domain_type;

    factor_estimator() { }
    virtual ~factor_estimator() { }
    virtual F operator()(const domain_type& args) const = 0;
  };

}

#endif
