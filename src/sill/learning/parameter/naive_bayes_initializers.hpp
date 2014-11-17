#ifndef SILL_NAIVE_BAYES_INITIALIZERS_HPP
#define SILL_NAIVE_BAYES_INITIALIZERS_HPP

#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/model/naive_bayes.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  inline void initialize(finite_variable* label,
                         const finite_domain& features,
                         const finite_dataset& /* ds */,
                         const factor_mle<table_factor>::param_type& /* params */,
                         unsigned seed,
                         naive_bayes<table_factor>& model) {
    // initialize the prior to the uniform distribution
    //table_factor prior(make_domain(label), 1.0 / label->size());
    //model = naive_bayes<table_factor>(prior);

    boost::mt19937 rng(seed);
    uniform_factor_generator gen;
    model = naive_bayes<table_factor>(gen(make_domain(label), rng));
    
    // initialize the feature CPDs uniformly
    foreach(finite_variable* feature, features) {
      model.add_feature(gen(make_domain(feature), make_domain(label), rng));
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
