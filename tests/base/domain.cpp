#define BOOST_TEST_MODULE domain
#include <boost/test/unit_test.hpp>

#include <sill/base/finite_variable.hpp>
#include <sill/base/universe.hpp>

using namespace sill;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;

  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  finite_var_vector vec;
  vec.push_back(x);
  vec.push_back(y);

  finite_domain empty;
  BOOST_CHECK_EQUAL(empty, make_domain<finite_variable>());

//   finite_domain one(x);
//   BOOST_CHECK_EQUAL(one, make_domain(x));

//   finite_domain two(vec);
//   BOOST_CHECK_EQUAL(two, make_domain(x, y));
}

BOOST_AUTO_TEST_CASE(test_operations) {
  universe u;
  
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  finite_variable* z = u.new_finite_variable("z", 4);
  finite_variable* w = u.new_finite_variable("w", 3);

  finite_domain xyz  = make_domain(x, y, z);
  finite_domain x1   = make_domain(x);
  finite_domain xy   = make_domain(x, y);
  finite_domain yw   = make_domain(y, w);
  finite_domain zw   = make_domain(z, w);
  finite_domain yz   = make_domain(y, z);
  finite_domain yzw  = make_domain(y, z, w);
  finite_domain xyzw = make_domain(x, y, z, w);
  
  BOOST_CHECK(includes(xyz, xy));
  BOOST_CHECK(!includes(xyz, yw));

  finite_domain intersection;
  finite_domain difference;
  //xyz.partition(yzw, intersection, difference);
  boost::tie(intersection, difference) = set_partition(xyz, yzw);
  BOOST_CHECK_EQUAL(intersection, yz);
  BOOST_CHECK_EQUAL(difference, x1);
  
  BOOST_CHECK_EQUAL(set_union(xyz, yzw), xyzw);
  BOOST_CHECK_EQUAL(set_union(xyz, x), xyz);
  BOOST_CHECK_EQUAL(set_intersect(xyz, yzw), yz);
  BOOST_CHECK_EQUAL(set_difference(xyz, yzw), x1);

  BOOST_CHECK(!set_disjoint(xyz, yw));
  BOOST_CHECK(set_disjoint(xy, zw));
}
