#define BOOST_TEST_MODULE uniform_factor_generator
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

size_t nsamples = 100;
const double lower = -0.7;
const double upper = +0.5;
const double exp_lower = std::exp(lower);
const double exp_upper = std::exp(upper);

BOOST_AUTO_TEST_CASE(test_all) {
  universe u;
  finite_variable* x = u.new_finite_variable(4);
  finite_variable* y = u.new_finite_variable(3);
  finite_domain xs = make_domain(x);
  finite_domain ys = make_domain(y);
  finite_domain xy = make_domain(x, y);
  
  boost::mt19937 rng;
  uniform_factor_generator gen(lower, upper);

  // check the marginals
  double sum = 0.0;
  for (size_t i = 0; i < nsamples; ++i) {
    table_factor f = gen(xy, rng);
    foreach(double x, f.values()) {
      sum += std::log(x);
      BOOST_CHECK(x >= exp_lower && x <= exp_upper);
    }
  }
  sum /= nsamples * num_assignments(xy);
  BOOST_CHECK_CLOSE_FRACTION(sum, (lower+upper)/2, 0.05);

  // check the conditionals
  for (size_t i = 0; i < nsamples; ++i) {
    table_factor f = gen(ys, xs, rng);
    BOOST_CHECK(f.is_conditional(xs));
  }
}
