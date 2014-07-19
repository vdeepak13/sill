#ifndef SILL_FACTOR_LEARNER_HPP
#define SILL_FACTOR_LEARNER_HPP

namespace sill {

  template <typename F>
  class factor_learner {
  public:
    typedef typename F::domain_type domain_type;

    factor_learner() { }
    virtual ~factor_learner() { }
    virtual F operator()(const domain_type& args) const = 0;
  };

}

#endif
