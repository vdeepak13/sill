#define BOOST_TEST_MODULE probability_array_mle
#include <boost/test/unit_test.hpp>

#include <sill/math/likelihood/probability_array_mle.hpp>

#include <sill/math/likelihood/probability_array_ll.hpp>
#include <sill/math/random/array_distribution.hpp>

#include <random>
#include <vector>

namespace sill {
  template class probability_array_mle<double, 1>;
  template class probability_array_mle<double, 2>;
  template class probability_array_mle<float, 1>;
  template class probability_array_mle<float, 2>;
}

using namespace sill;

size_t nsamples = 10000;
double tol = 0.01;

template <size_t N, typename Array>
double reconstruction_error(const Array& param) {
  // generate a few samples
  std::mt19937 rng;
  array_distribution<double, N> dist(param);
  typedef typename array_distribution<double, N>::result_type sample_type;
  std::vector<std::pair<sample_type, double>> samples;
  samples.reserve(nsamples);
  for (size_t i = 0; i < nsamples; ++i) {
    samples.emplace_back(dist(rng), 1.0);
  }

  // compute the MLE and compare against ground truth
  probability_array_mle<double, N> mle;
  Array estim(param.rows(), param.cols());
  mle.estimate(samples, estim);

  double ll_truth = probability_array_ll<double, N>(param).log(samples);
  double ll_estim = probability_array_ll<double, N>(estim).log(samples);
  std::cout << "Log-likelihood of the original: " << ll_truth << std::endl;
  std::cout << "Log-likelihood of the estimate: " << ll_estim << std::endl;
  BOOST_CHECK_CLOSE(ll_truth, ll_estim, 1.0);

  return abs(estim-param).maxCoeff();
}

BOOST_AUTO_TEST_CASE(test_mle1) {
  Eigen::Array<double, Eigen::Dynamic, 1> param(4);
  param << 0.1, 0.4, 0.3, 0.2;
  BOOST_CHECK_SMALL(reconstruction_error<1>(param), tol);
}

BOOST_AUTO_TEST_CASE(test_mle2) {
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> param(2, 3);
  param << 0.1, 0.2, 0.05, 0.4, 0.05, 0.2;
  BOOST_CHECK_SMALL(reconstruction_error<2>(param), tol);
}
