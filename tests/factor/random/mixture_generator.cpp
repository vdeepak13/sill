#define BOOST_TEST_MODULE mixture_generator
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/mixture_generator.hpp>
#include <sill/factor/random/moment_gaussian_generator.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  template class mixture_generator<moment_gaussian_generator>;
}

using namespace sill;

BOOST_AUTO_TEST_CASE(test_all) {
  universe u;
  vector_variable* x = u.new_vector_variable(2);
  vector_variable* y = u.new_vector_variable(1);
  vector_domain xs = make_domain(x);
  vector_domain ys = make_domain(y);
  vector_domain xy = make_domain(x, y);
  
  moment_gaussian_generator base_gen(2.0);

  boost::mt19937 rng;
  mixture_generator<moment_gaussian_generator> gen1;
  mixture_generator<moment_gaussian_generator> gen2(2);
  mixture_generator<moment_gaussian_generator> gen3(2, base_gen);
  mixture_generator<moment_gaussian_generator> gen4(2, base_gen.param());

  // check the constructors
  BOOST_CHECK_EQUAL(gen1(xy, rng).size(), 3);
  BOOST_CHECK_EQUAL(gen2(xy, rng).size(), 2);
  BOOST_CHECK_EQUAL(gen3.param().k, 2);
  BOOST_CHECK_EQUAL(gen3.param().base_params.mean_lower, 2.0);
  BOOST_CHECK_EQUAL(gen4.param().k, 2);
  BOOST_CHECK_EQUAL(gen4.param().base_params.mean_lower, 2.0);

  // check the marginal distribution
  mixture<moment_gaussian> f = gen1(xy, rng);
  BOOST_CHECK_EQUAL(f.size(), 3);
  BOOST_CHECK_EQUAL(f.arguments(), xy);
  
  // check the conditional distribution
  mixture<moment_gaussian> g = gen1(ys, xs, rng);
  BOOST_CHECK_EQUAL(g.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(g[i].head(), make_vector(y));
    BOOST_CHECK_EQUAL(g[i].tail(), make_vector(x));
  }
}

