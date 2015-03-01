#define BOOST_TEST_MODULE array_domain
#include <boost/test/unit_test.hpp>

#include <sill/argument/array_domain.hpp>

namespace sill {
  template class array_domain<int, 2>;
  template class array_domain<int, 9>;
}

typedef sill::array_domain<int, 1> domain1;
typedef sill::array_domain<int, 2> domain2;
typedef sill::array_domain<int, 3> domain3;

BOOST_AUTO_TEST_CASE(test_constructors) {
  domain2 a = {0, 0};
  domain2 b = {1, 3};
  BOOST_CHECK(!b.empty());
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b[0], 1);
  BOOST_CHECK_EQUAL(b[1], 3);

  using std::swap;
  swap(a, b);
  BOOST_CHECK_EQUAL(b[0], 0);
  BOOST_CHECK_EQUAL(a.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_elems) {
  domain1 a = { 3 };
  BOOST_CHECK_EQUAL(a.count(3), 1);
  BOOST_CHECK_EQUAL(a.count(5), 0);
}

BOOST_AUTO_TEST_CASE(test_operations) {
  domain2 a = {3, 5};
  domain1 b = {5};
  domain1 c = {2};
  domain1 d = {3};
  domain2 ar = {5, 3};
  BOOST_CHECK_EQUAL(a + c, domain3({3, 5, 2}));
  BOOST_CHECK_EQUAL(a - b, domain1({3}));
  BOOST_CHECK_EQUAL(a & b, domain1({5}));
  BOOST_CHECK_EQUAL(b | c, domain2({5, 2}));
  BOOST_CHECK(!disjoint(a, b));
  BOOST_CHECK(disjoint(a, c));
  BOOST_CHECK(!equivalent(a, b));
  BOOST_CHECK(!equivalent(a, c));
  BOOST_CHECK(equivalent(a, ar));
  BOOST_CHECK(!subset(a, b));
  BOOST_CHECK(subset(d, a));
  BOOST_CHECK(superset(a, b));
  BOOST_CHECK(superset(a, d));
}

BOOST_AUTO_TEST_CASE(test_equivalent2) {
  domain2 a = {3, 5};
  domain2 c = {2, 5};
  domain2 d = {5, 3};
  BOOST_CHECK(!equivalent(a, c));
  BOOST_CHECK(equivalent(a, d));
}