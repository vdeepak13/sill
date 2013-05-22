#define BOOST_TEST_MODULE variable
#include <boost/test/unit_test.hpp>

#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/universe.hpp>

using namespace sill;

BOOST_AUTO_TEST_CASE(test_constructors) {
  // Create a universe.
  universe u;

  // Create some variables.
  finite_variable* x = u.new_finite_variable("x", 3);
  vector_variable* y = u.new_vector_variable("y", 2);
  finite_variable* z = u.new_finite_variable(2);
  vector_variable* q = u.new_vector_variable(2);

  BOOST_CHECK_EQUAL(x->name(), "x");
  BOOST_CHECK_EQUAL(x->size(), 3);
  BOOST_CHECK_EQUAL(x->id(), 0); // this is undesirable--variables should start at 1

  BOOST_CHECK_EQUAL(y->name(), "y");
  BOOST_CHECK_EQUAL(y->size(), 2);
  BOOST_CHECK_EQUAL(y->id(), 1);

  BOOST_CHECK_EQUAL(z->size(), 2);
  BOOST_CHECK_EQUAL(z->id(), 2);

  BOOST_CHECK_EQUAL(q->size(), 2);
  BOOST_CHECK_EQUAL(q->id(), 3);
}
