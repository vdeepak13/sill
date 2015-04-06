#define BOOST_TEST_MODULE probability_table_mle
#include <boost/test/unit_test.hpp>

#include <sill/math/likelihood/probability_table_mle.hpp>

#include <sill/math/likelihood/probability_table_ll.hpp>
#include <sill/math/random/table_distribution.hpp>

#include <random>
#include <vector>

namespace sill {
  template class probability_table_mle<double>;
  template class probability_table_mle<float>;
}

using namespace sill;

size_t nsamples = 10000;
double tol = 0.01;
                    
BOOST_AUTO_TEST_CASE(test_mle) {
  table<double> param({2, 3}, {0.1, 0.05, 0.15, 0.25, 0.2, 0.25});

  // generate a few samples
  std::mt19937 rng;
  table_distribution<double> dist(param);
  std::vector<std::pair<finite_index, double>> samples;
  samples.reserve(nsamples);
  for (size_t i = 0; i < nsamples; ++i) {
    samples.emplace_back(dist(rng), 1.0);
  }

  // compute the MLE and compare against ground truth
  probability_table_mle<double> mle;
  table<double> estim(param.shape());
  mle.estimate(samples, estim);
  double diff =
    std::inner_product(estim.begin(), estim.end(), param.begin(),
                       0.0, maximum<double>(), abs_difference<double>());
  BOOST_CHECK_SMALL(diff, tol);

  double ll_truth = probability_table_ll<double>(param).log(samples);
  double ll_estim = probability_table_ll<double>(estim).log(samples);
  std::cout << "Log-likelihood of the original: " << ll_truth << std::endl;
  std::cout << "Log-likelihood of the estimate: " << ll_estim << std::endl;
  BOOST_CHECK_CLOSE(ll_truth, ll_estim, 1.0);
}
