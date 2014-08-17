#ifndef SILL_FACTOR_RANDOM_ADAPTORS_HPP
#define SILL_FACTOR_RANDOM_ADAPTORS_HPP

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/ref.hpp>

namespace sill {

  template <typename Gen, typename Engine>
  boost::function<
    typename Gen::result_type(const typename Gen::domain_type&)>
  marginal_fn(Gen gen, Engine& engine) {
    return boost::bind(gen, _1, boost::ref(engine));
  }
  
  template <typename Gen, typename Engine>
  boost::function<
    typename Gen::result_type(const typename Gen::domain_type&,
                              const typename Gen::domain_type&)>
  conditional_fn(Gen gen, Engine& engine) {
    return boost::bind(gen, _1, _2, boost::ref(engine));
  }

  template <typename Gen, typename Engine>
  boost::function<
    typename Gen::result_type(const typename Gen::output_domain_type&,
                              const typename Gen::input_domain_type&)>
  crf_factor_fn(Gen gen, Engine& engine) {
    return boost::bind(gen, _1, _2, boost::ref(engine));
  }

}

#endif
