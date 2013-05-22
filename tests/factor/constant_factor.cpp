#define BOOST_TEST_MODULE constant_factor
#include <boost/test/unit_test.hpp>

#include <sill/factor/constant_factor.hpp>

#include "predicates.hpp"

using namespace sill;

BOOST_AUTO_TEST_CASE(test_operations) {
  constant_factor f(1.0);
  constant_factor g(2.0);
  constant_factor fg = combine(f, g, product_op);
  constant_factor h = fg.marginal(finite_domain());

  BOOST_CHECK_EQUAL(fg.value, 2.0);
  BOOST_CHECK_EQUAL(fg.value, h.value);
}

BOOST_AUTO_TEST_CASE(test_serialization) {
  universe u;
  BOOST_CHECK(serialize_deserialize(constant_factor(2.0), u));
}
