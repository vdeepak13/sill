#define BOOST_TEST_MODULE associative_factor_generator
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/associative_factor_generator.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

size_t nsamples = 1000;
const double lower = -0.7;
const double upper = +0.5;
const double exp_lower = std::exp(lower);
const double exp_upper = std::exp(upper);

BOOST_AUTO_TEST_CASE(test_associative) {
  universe u;
  finite_variable* x = u.new_finite_variable(3);
  finite_variable* y = u.new_finite_variable(3);
  finite_domain xs = make_domain(x);
  finite_domain ys = make_domain(y);
  finite_domain xy = make_domain(x, y);

  boost::mt19937 rng;
  associative_factor_generator gen(lower, upper);

  // check the marginals
  double sum = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < nsamples; ++i) {
    table_factor f = gen(xy, rng);
    foreach(finite_assignment a, assignments(xy)) {
      if (a[x] == a[y]) {
        BOOST_CHECK(f(a) >= exp_lower && f(a) <= exp_upper);
        sum += std::log(f(a));
        ++count;
      } else {
        BOOST_CHECK_EQUAL(f(a), 1.0);
      }
    }
  }
  BOOST_CHECK_CLOSE_FRACTION(sum / count, (lower+upper)/2, 0.05);
}
