#define BOOST_TEST_MODULE multinomial_distribution
#include <boost/test/unit_test.hpp>

#include <sill/math/random/multinomial_distribution.hpp>

#include <random>

using namespace sill;

BOOST_AUTO_TEST_CASE(test_sampling) {
  std::mt19937 rng;

  std::vector<double> p = {0.2, 0.2, 0.5, 0.1};
  std::vector<double> count(4);
  multinomial_distribution<double> dist(p);

  size_t nsamples = 200000;
  for(size_t i = 0; i < nsamples; i++) {
    count[dist(rng)] += 1.0 / nsamples;
  }
  
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(p[i], count[i], 1);
  }
}
