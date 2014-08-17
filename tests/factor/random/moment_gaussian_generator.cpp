#define BOOST_TEST_MODULE moment_gaussian_generator
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/moment_gaussian_generator.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

size_t nsamples = 100;

BOOST_AUTO_TEST_CASE(test_all) {
  universe u;
  vector_variable* x1 = u.new_vector_variable(1);
  vector_variable* x2 = u.new_vector_variable(2);
  vector_variable* y = u.new_vector_variable(1);
  vector_domain xs = make_domain(x1, x2);
  vector_domain ys = make_domain(y);
  vector_domain xy = make_domain(x1, x2, y);

  boost::mt19937 rng;
  moment_gaussian_generator gen(-0.5, 1.5, 2.0, 0.3, 0.0);
  
  // test marginals
  double sum = 0.0;
  for (size_t i = 0; i < nsamples; ++i) {
    moment_gaussian mg = gen(xy, rng);
    const vec& mean = mg.mean();
    const mat& cov = mg.covariance();
    BOOST_CHECK(mg.marginal());
    BOOST_CHECK_EQUAL(mean.size(), 4);
    BOOST_CHECK_EQUAL(cov.n_rows, 4);
    BOOST_CHECK_EQUAL(cov.n_cols, 4);
    foreach(double x, mean) {
      BOOST_CHECK(-0.5 <= x && x <= 1.5);
      sum += x;
    }
    for (size_t r = 0; r < 4; ++r) {
      for (size_t c = 0; c < 4; ++c) {
        if (r == c) {
          BOOST_CHECK_CLOSE(cov(r, c), 2.0, 1e-10);
        } else {
          BOOST_CHECK_CLOSE(cov(r, c), 0.6, 1e-10);
        }
      }
    }
  }
  BOOST_CHECK_CLOSE_FRACTION(sum / nsamples / 4, 0.5, 0.05);
  
  // test conditionals
  double sum_mean = 0.0;
  double sum_coef = 0.0;
  for (size_t i = 0; i < nsamples; ++i) {
    moment_gaussian mg = gen(ys, xs, rng);
    const vec& mean = mg.mean();
    const mat& cov  = mg.covariance();
    const mat& coef = mg.coefficients();
    BOOST_CHECK(!mg.marginal());
    BOOST_CHECK_EQUAL(mg.size_head(), 1);
    BOOST_CHECK_EQUAL(mg.size_tail(), 3);
    foreach(double x, mean) {
      BOOST_CHECK(-0.5 <= x && x <= 1.5);
      sum_mean += x;
    }
    foreach(double x, coef) {
      BOOST_CHECK(0.0 <= x && x <= 1.0);
      sum_coef += x;
    }
  }
  BOOST_CHECK_CLOSE_FRACTION(sum_mean / nsamples, 0.5, 0.05);
  BOOST_CHECK_CLOSE_FRACTION(sum_coef / nsamples / 3, 0.5, 0.05);
}
