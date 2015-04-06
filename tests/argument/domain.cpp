#define BOOST_TEST_MODULE domain
#include <boost/test/unit_test.hpp>

#include <sill/argument/domain.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/universe.hpp>

namespace sill {
  template class domain<finite_variable*>;
  template class domain<vector_variable*>;
  template class domain<size_t>;
}

using namespace sill;

typedef domain<finite_variable*> domain_type;

BOOST_AUTO_TEST_CASE(test_constructors) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  domain_type a;
  BOOST_CHECK(a.empty());

  domain_type b({x, y});
  BOOST_CHECK_EQUAL(b.size(), 2);
  BOOST_CHECK_EQUAL(b[0], x);
  BOOST_CHECK_EQUAL(b[1], y);

  domain_type c(std::vector<finite_variable*>(1, x));
  BOOST_CHECK_EQUAL(c.size(), 1);
  BOOST_CHECK_EQUAL(c[0], x);
}

BOOST_AUTO_TEST_CASE(test_operations) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  finite_variable* z = u.new_finite_variable("z", 4);
  finite_variable* w = u.new_finite_variable("w", 3);

  domain_type xyz  = {x, y, z};
  domain_type x1   = {x};
  domain_type y1   = {y};
  domain_type z1   = {z};
  domain_type xy   = {x, y};
  domain_type xw   = {x, w};
  domain_type yx   = {y, x};
  domain_type yw   = {y, w};
  domain_type yz   = {y, z};
  domain_type zw   = {z, w};
  domain_type xyw  = {x, y, w};
  domain_type yzw  = {y, z, w};
  domain_type xyzw = {x, y, z, w};
  domain_type xwzy = {x, w, z, y};
  domain_type xywx = {x, y, w, x};

  BOOST_CHECK_EQUAL(x1 + y1, xy);
  BOOST_CHECK_EQUAL(xy + z1, xyz);
  BOOST_CHECK_EQUAL(xy - z1, xy);
  BOOST_CHECK_EQUAL(xy - yz, x1);
  BOOST_CHECK_EQUAL(xy | z1, xyz);
  BOOST_CHECK_EQUAL(xy | yw, xyw);
  BOOST_CHECK_EQUAL(xy & yw, y1);
  BOOST_CHECK(disjoint(xy, z1));
  BOOST_CHECK(!disjoint(xy, yzw));
  BOOST_CHECK(equivalent(xy, yx));
  BOOST_CHECK(!equivalent(yw, zw));
  BOOST_CHECK(subset(yx, xyz));
  BOOST_CHECK(!subset(yx, yw));
  BOOST_CHECK(superset(xyzw, yx));
  BOOST_CHECK(!superset(xyw, xyz));
  BOOST_CHECK(type_compatible(xy, xw));
  BOOST_CHECK(!type_compatible(xyz, xyw));
  BOOST_CHECK(!type_compatible(x1, y1));

  BOOST_CHECK_EQUAL(xyz.count(x), 1);
  BOOST_CHECK_EQUAL(xyz.count(w), 0);
  xywx.unique();
  BOOST_CHECK_EQUAL(xywx.size(), 3);
  BOOST_CHECK(equivalent(xywx, xyw));
  
  finite_var_map map;
  map[x] = x;
  map[y] = w;
  map[z] = z;
  map[w] = y;
  xyzw.subst(map);
  BOOST_CHECK_EQUAL(xyzw, xwzy);
}
