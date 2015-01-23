#define BOOST_TEST_MODULE functions
#include <boost/test/unit_test.hpp>

#include <sill/math/function/logistic_discrete.hpp>
#include <sill/math/function/softmax.hpp>

BOOST_AUTO_TEST_CASE(test_logistic_discrete) {
  sill::logistic_discrete f("1 2 3; 4 5 6", 1);
  BOOST_CHECK_CLOSE(f(arma::uvec("0 2")), 1.0/(1.0 + exp(-8.0)), 1e-10);
}

BOOST_AUTO_TEST_CASE(test_softmax) {
  sill::softmax f("1;-1", "1 1");
  arma::vec result = f("1");
  BOOST_CHECK_CLOSE(result[0], 1.0 / (1 + exp(-2.0)), 1e-10);
  BOOST_CHECK_CLOSE(result[1], 1.0 / (1 + exp(+2.0)), 1e-10);
}
