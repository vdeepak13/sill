#ifndef SILL_NAIVE_BAYES_INIT_HPP
#define SILL_NAIVE_BAYES_INIT_HPP

#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/factor/hybrid.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/model/naive_bayes.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Randomly initializes a naive Bayes model for table factors.
   * Both the prior and the CPDs are drawn from uniform_factor_generator.
   */
  inline void initialize(finite_variable* label,
                         const finite_var_vector& features,
                         const finite_dataset& /* ds */,
                         const factor_mle<table_factor>::param_type& /* params */,
                         unsigned seed,
                         naive_bayes<table_factor>& model) {
    // initialize the prior
    boost::mt19937 rng(seed);
    uniform_factor_generator gen;
    model = naive_bayes<table_factor>(gen(make_domain(label), rng));
    
    // initialize the feature CPDs
    foreach(finite_variable* feature, features) {
      model.add_feature(gen(make_domain(feature), make_domain(label), rng));
    }
  }

  /**
   * Randomly initializes a naive Bayes model for hybrid factors.
   * Both the prior and the discrete CPDs are drawn from 
   * uniform_factor_generator. Gaussian CPDs are drawn from data.
   */
  inline void initialize(finite_variable* label,
                         const var_vector& features,
                         const hybrid_dataset<>& ds,
                         const factor_mle<hybrid<moment_gaussian> >::param_type& params,
                         unsigned seed,
                         naive_bayes<hybrid<moment_gaussian> >& model) {
    typedef hybrid<moment_gaussian> hybrid_moment;

    // initialize the prior
    boost::mt19937 rng(seed);
    uniform_factor_generator gen;
    model = naive_bayes<hybrid_moment>(gen(make_domain(label), rng));
    
    // initialize the feature CPDs
    factor_mle<moment_gaussian> mle(&ds, params.comp_params);
    foreach(variable* feature, features) {
      switch (feature->type()) {
      case variable::FINITE_VARIABLE: {
        finite_variable* var = dynamic_cast<finite_variable*>(feature);
        table_factor f = gen(make_domain(var), make_domain(label), rng);
        model.add_feature(hybrid_moment(f));
        break;
      }
      case variable::VECTOR_VARIABLE: {
        vector_variable* var = dynamic_cast<vector_variable*>(feature);
        moment_gaussian f = mle(make_domain(var));
        hybrid_moment h(make_vector(label), make_vector(var));
        for (size_t i = 0; i < h.size(); ++i) {
          h[i].mean() = ds.sample(make_vector(var), rng).values;
          h[i].covariance() = f.covariance();
        }
        model.add_feature(h);
        break;
      }
      }
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
