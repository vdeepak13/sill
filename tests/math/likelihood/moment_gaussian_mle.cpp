#define BOOST_TEST_MODULE moment_gaussian_mle
#include <boost/test/unit_test.hpp>

#include <sill/math/likelihood/moment_gaussian_mle.hpp>
#include <sill/math/random/gaussian_distribution.hpp>

#include <random>
#include <vector>

namespace sill {
  template class moment_gaussian_mle<double>;
  template class moment_gaussian_mle<float>;
}

using namespace sill;

size_t nsamples = 10000;
double tol = 0.05;

BOOST_AUTO_TEST_CASE(test_mle) {
  typedef dynamic_vector<double> vec_type;
  moment_gaussian_param<double> param(3, 0);
  param.mean << 1.0, 3.0, 2.0;
  param.cov  << 3.0, 2.0, 1.0,
                2.0, 1.5, 1.0,
                1.0, 1.0, 1.5;

  // generate a few samples
  std::mt19937 rng;
  gaussian_distribution<double> dist(param);
  std::vector<std::pair<vec_type, double>> samples;
  samples.reserve(nsamples);
  for (size_t i = 0; i < nsamples; ++i) {
    samples.emplace_back(dist(rng), 1.0);
  }

  // compute the MLE and compare against ground truth
  moment_gaussian_mle<double> mle;
  moment_gaussian_param<double> estim(3, 0);
  mle.estimate(samples, estim);
  BOOST_CHECK_SMALL((param.mean - estim.mean).cwiseAbs().maxCoeff(), tol);
  BOOST_CHECK_SMALL((param.cov - estim.cov).cwiseAbs().maxCoeff(), tol);
}
