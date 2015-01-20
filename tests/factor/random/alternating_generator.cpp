#define BOOST_TEST_MODULE alternating_generator
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/alternating_generator.hpp>
#include <sill/factor/random/moment_gaussian_generator.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>

#include <boost/random/mersenne_twister.hpp>

namespace sill{
  template class alternating_generator<uniform_factor_generator>;
  template class alternating_generator<moment_gaussian_generator>;
}
using namespace sill;

BOOST_AUTO_TEST_CASE(test_constructors) {
  uniform_factor_generator def_gen(1.0, 2.0);
  uniform_factor_generator alt_gen(3.0, 4.0);
  uniform_factor_generator::param_type def_par = def_gen.param();
  uniform_factor_generator::param_type alt_par = alt_gen.param();
  
  
  alternating_generator<uniform_factor_generator> gen1(def_gen, alt_gen, 1);
  alternating_generator<uniform_factor_generator> gen2(def_par, alt_par, 3);

  BOOST_CHECK_EQUAL(gen1.param().def_params.lower, 1.0);
  BOOST_CHECK_EQUAL(gen1.param().alt_params.lower, 3.0);
  BOOST_CHECK_EQUAL(gen2.param().def_params.lower, 1.0);
  BOOST_CHECK_EQUAL(gen2.param().alt_params.lower, 3.0);
  BOOST_CHECK_EQUAL(gen1.param().period, 1);
  BOOST_CHECK_EQUAL(gen2.param().period, 3);
}

BOOST_AUTO_TEST_CASE(test_operators) {
  universe u;
  finite_variable* x = u.new_finite_variable(2);
  finite_variable* y = u.new_finite_variable(1);
  finite_domain xs = make_domain(x);
  finite_domain ys = make_domain(y);
  finite_domain xy = make_domain(x, y);

  uniform_factor_generator def_gen(-2.0, -1.0); // log space
  uniform_factor_generator alt_gen(+1.0, +2.0); // log space
  
  alternating_generator<uniform_factor_generator> gen(def_gen, alt_gen, 3);
  boost::mt19937 rng;

  // test marginals
  BOOST_CHECK_LT(*gen(xy, rng).begin(), 1.0);
  BOOST_CHECK_LT(*gen(xy, rng).begin(), 1.0);
  BOOST_CHECK_GT(*gen(xy, rng).begin(), 1.0);
  BOOST_CHECK_LT(*gen(xy, rng).begin(), 1.0);
  BOOST_CHECK_LT(*gen(xy, rng).begin(), 1.0);
  BOOST_CHECK_GT(*gen(xy, rng).begin(), 1.0);

  // test conditionals
  // not sure what to do besides instantiating the operator 
  gen(ys, xs, rng);
  gen(ys, xs, rng);
  gen(ys, xs, rng);
  gen(ys, xs, rng);
}
