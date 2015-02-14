#define BOOST_TEST_MODULE bounded_domain
#include <boost/test/unit_test.hpp>

#include <sill/base/bounded_domain.hpp>

namespace sill {
  template class bounded_domain<int, 2>;
  template class bounded_domain<int, 9>;
}

typedef sill::bounded_domain<int, 5> domain5;
typedef sill::bounded_domain<int, 2> domain2;

BOOST_AUTO_TEST_CASE(test_constructors) {
  domain5 a;
  BOOST_CHECK(a.empty());
  
  domain5 b = {1, 3};
  BOOST_CHECK(!b.empty());
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b[0], 1);
  BOOST_CHECK_EQUAL(b[1], 3);

  using std::swap;
  swap(a, b);
  BOOST_CHECK(b.empty());
  BOOST_CHECK_EQUAL(a.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_elems) {
  domain5 a = { 3 };
  BOOST_CHECK_EQUAL(a.count(3), 1);
  BOOST_CHECK_EQUAL(a.count(5), 0);

  a.push_back(2);
  BOOST_CHECK_EQUAL(a.size(), 2);
  BOOST_CHECK_EQUAL(a.count(2), 1);

  BOOST_CHECK(a.insert(10).second);
  BOOST_CHECK(!a.insert(2).second);
  BOOST_CHECK_EQUAL(a.size(), 3);
  BOOST_CHECK_EQUAL(a[0], 3);
  BOOST_CHECK_EQUAL(a[1], 2);
  BOOST_CHECK_EQUAL(a[2], 10);

  BOOST_CHECK_EQUAL(a.erase(3), 1);
  BOOST_CHECK_EQUAL(a.size(), 2);
  BOOST_CHECK_EQUAL(a[0], 2);
  BOOST_CHECK_EQUAL(a[1], 10);
  
  BOOST_CHECK_EQUAL(a.erase(1), 0);
  BOOST_CHECK_EQUAL(a.size(), 2);

  a.clear();
  BOOST_CHECK(a.empty());
  BOOST_CHECK_EQUAL(a.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_operations) {
  domain5 a = {3, 5};
  domain5 b = {5, 1};
  domain5 c = {2};
  domain5 d = {3};
  domain5 ar = {5, 3};
  BOOST_CHECK_EQUAL(left_union(a, b), domain5({3, 5, 1}));
  BOOST_CHECK_EQUAL(right_union(a, b), domain5({5, 1, 3}));
  BOOST_CHECK_EQUAL(concat(a, c), domain5({3, 5, 2}));
  BOOST_CHECK_EQUAL(intersection(a, b), domain5({5}));
  BOOST_CHECK_EQUAL(difference(a, b), domain5({3}));
  BOOST_CHECK_EQUAL(difference(a, c), domain5({3, 5}));
  BOOST_CHECK_EQUAL(partition(a, b).first, domain5({5}));
  BOOST_CHECK_EQUAL(partition(a, b).second, domain5({3}));
  BOOST_CHECK(!disjoint(a, b));
  BOOST_CHECK(disjoint(a, c));
  BOOST_CHECK(!equivalent(a, b));
  BOOST_CHECK(!equivalent(a, c));
  BOOST_CHECK(equivalent(a, ar));
  BOOST_CHECK(!subset(a, b));
  BOOST_CHECK(subset(d, a));
  BOOST_CHECK(!superset(a, b));
  BOOST_CHECK(superset(a, d));
}

BOOST_AUTO_TEST_CASE(test_equivalent2) {
  domain2 a = {3, 5};
  domain2 b = {3};
  domain2 c = {2, 5};
  domain2 d = {5, 3};
  BOOST_CHECK(!equivalent(a, b));
  BOOST_CHECK(!equivalent(a, c));
  BOOST_CHECK(equivalent(a, d));
}
