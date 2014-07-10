#define BOOST_TEST_MODULE multinomial_distribution

#include <boost/random/mersenne_twister.hpp>
#include <boost/test/unit_test.hpp>

#include <sill/math/multinomial_distribution.hpp>

BOOST_AUTO_TEST_CASE(test_sampling) {
  boost::mt19937 rng;

  sill::multinomial_distribution dist("0.2 0.2 0.5 0.1");
  arma::vec count = arma::zeros(4);

  for(size_t i = 0; i < 100000; i++) {
    count(dist(rng))++;
  }
  count /= sum(count);
  
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(dist.p()[i], count[i], 1.0 /* 1% */);
  }
}

