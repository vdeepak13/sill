#ifndef SILL_FACTOR_GENERATOR_HPP
#define SILL_FACTOR_GENERATOR_HPP

namespace sill {

  template <typename F>
  struct factor_generator {
    typedef F                         factor_type;
    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type   domain_type;

    //! Generates the marginal distribution over the given argument set
    virtual F operator()(const domain_type& args) = 0;

    //! Generates the marginal distribution over a single variable
    virtual F operator()(variable_type* x) {
      return operator()(make_domain(x));
    }

  };

} // namespace sill

#endif
