#ifndef SILL_MIXTURE_INIT_HPP
#define SILL_MIXTURE_INIT_HPP

#include <sill/factor/mixture.hpp>
#include <sill/factor/moment_gaussian.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Randomly initalizes a mixture Gaussian model.
   * Sets the mean of each component to a random data point and the covariance
   * to the covariance of the whole population.
   */
  inline void initialize(size_t num_components,
                         const vector_var_vector& args,
                         const vector_dataset<>& ds,
                         const factor_mle<moment_gaussian>::param_type& params,
                         unsigned seed,
                         mixture<moment_gaussian>& model) {
    assert(num_components > 0);
    assert(ds.size() >= num_components);

    // generate the row indices that will determine the centers
    // we want to be absolutely certain they are distinct.
    // otherwise, EM will not compute distinct clusters
    boost::mt19937 rng(seed);
    boost::uniform_int<size_t> uniform(0, ds.size() - 1);
    boost::unordered_set<size_t> random_rows;
    while (random_rows.size() < num_components) {
      random_rows.insert(uniform(rng));
    }

    // compute the covariance and set the random means
    factor_mle<moment_gaussian> estim(&ds, params);
    model = mixture_gaussian(num_components, estim(args));
    size_t k = 0;
    foreach(size_t row, random_rows) {
      model[k++].mean() = ds.record(row, args).values;
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
