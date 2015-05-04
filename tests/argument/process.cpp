#define BOOST_TEST_MODULE process
#include <boost/test/unit_test.hpp>

#include <sill/argument/universe.hpp>

using namespace sill;

struct fixture {
  fixture()
    : p(u.new_finite_dprocess("p", 4)),
      q(u.new_finite_dprocess("q", 2)) { }
  universe u;
  dprocess p;
  dprocess q;
};

BOOST_FIXTURE_TEST_CASE(test_construct, fixture) {
  BOOST_CHECK_EQUAL(p.name(), "p");
  BOOST_CHECK_EQUAL(p.size(), 4);
  BOOST_CHECK_EQUAL(q.name(), "q");
  BOOST_CHECK_EQUAL(q.size(), 2);
}

BOOST_FIXTURE_TEST_CASE(test_variables, fixture) {
  universe u;

  BOOST_CHECK_EQUAL(p(5).size(), 4);
  BOOST_CHECK_EQUAL(p(5).index(), 5);
  BOOST_CHECK_EQUAL(p(5).name(), "p");
  BOOST_CHECK_EQUAL(q(8).size(), 2);
  BOOST_CHECK_EQUAL(q(8).index(), 8);
  BOOST_CHECK_EQUAL(q(8).name(), "q");
}