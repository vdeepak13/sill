#define BOOST_TEST_MODULE uniform_factor_generator
#include <boost/test/unit_test.hpp>

#include <sill/factor/random/uniform_table_generator.hpp>

#include <sill/argument/universe.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/factor/probability_table.hpp>

#include <random>

#include <boost/mpl/list.hpp>

namespace sill {
  template class uniform_table_generator<ctable>;
  template class uniform_table_generator<ptable>;
}

using namespace sill;

size_t nsamples = 100;
const double lower = -0.7;
const double upper = +0.5;

typedef boost::mpl::list<ctable,ptable> factor_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_all, F, factor_types) {
  universe u;
  variable x = u.new_finite_variable("x", 4);
  variable y = u.new_finite_variable("y", 3);
  domain xs = { x };
  domain ys = { y };
  domain xy = { x, y };
  
  std::mt19937 rng;
  uniform_table_generator<F> gen(lower, upper);

  // check the marginals
  double sum = 0.0;
  for (size_t i = 0; i < nsamples; ++i) {
    F f = gen(xy, rng);
    for (double x : f.param()) {
      BOOST_CHECK(x >= lower && x <= upper);
      sum += x;
    }
  }
  sum /= nsamples * finite_size(xy);
  BOOST_CHECK_CLOSE_FRACTION(sum, (lower + upper) / 2, 0.05);

  /*
  // check the conditionals
  for (size_t i = 0; i < nsamples; ++i) {
    table_factor f = gen(ys, xs, rng);
    BOOST_CHECK(f.is_conditional(xs));
  }
  */
}
