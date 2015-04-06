#define BOOST_TEST_MODULE softmax_distribution
#include <boost/test/unit_test.hpp>

#include <sill/math/random/softmax_distribution.hpp>

namespace sill {
  template class softmax_distribution<double>;
  template class softmax_distribution<float>;
}

using namespace sill;

size_t nsamples = 5000;
double tol = 0.03;

dynamic_vector<double>
sample(const softmax_distribution<double>& d,
       const dynamic_vector<double>& tail) {
  std::mt19937 rng;
  dynamic_vector<double> result(d.param().num_labels());
  result.fill(0.0);
  for (size_t i = 0; i < nsamples; ++i) {
    ++result[d(rng, tail)];
  }
  result /= nsamples;
  return result;
}

BOOST_AUTO_TEST_CASE(test_conditional) {
  softmax_param<double> param(3, 2);
  param.bias() << 0.1, 0.2, 0.3;
  param.weight() << -0.1, 0.1, 0.2, 0.3, -0.2, 0.3;
  softmax_distribution<double> distribution(param);
  for (double a = -2.0; a <= 2.0; a += 0.5) {
    double b = 1.0;
    dynamic_vector<double> tail(2);
    tail[0] = a;
    tail[1] = b;
    dynamic_vector<double> result(3);
    result[0] = std::exp(0.1 - 0.1*a + 0.1*b);
    result[1] = std::exp(0.2 + 0.2*a + 0.3*b);
    result[2] = std::exp(0.3 - 0.2*a + 0.3*b);
    result /= result.sum();
    dynamic_vector<double> estimate = sample(distribution, tail);
    BOOST_CHECK_SMALL((result - estimate).cwiseAbs().sum(), tol);
  }
}
