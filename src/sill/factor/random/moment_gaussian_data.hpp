#ifndef SILL_MOMENT_GAUSSIAN_DATA_HPP
#define SILL_MOMENT_GAUSSIAN_DATA_HPP

#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/random/factor_generator.hpp>
#include <sill/learning/dataset2/vector_dataset.hpp>
#include <sill/learning/parameter/moment_gaussian_mle.hpp>

namespace sill {

  /**
   * A moment Gaussian generator that draws the means directly from
   * a dataset and sets the covariance to the empirical covariance.
   * Used primarily in mixture_em.
   * TODO: add caching of the mle query
   */
  template <typename Dataset = vector_dataset<> >
  class moment_gaussian_data : factor_generator<moment_gaussian> {
  public:
    moment_gaussian_data(const Dataset* dataset, double smoothing = 0.0)
      : dataset(dataset), mle(dataset, smoothing) {
    }

    moment_gaussian operator()(const vector_domain& args) {
      moment_gaussian factor = mle(args);
      factor.mean() = dataset->sample(factor.head(), rng).values;
      return factor;
    }
    
  private:
    const Dataset* dataset;
    moment_gaussian_mle<Dataset> mle;
    boost::mt19937 rng;
  };

}  // namespace sill

#endif

