#define BOOST_TEST_MODULE array_factor
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/base/array_factor.hpp>
#include <sill/functional/operators.hpp>
#include <sill/functional/eigen.hpp>

namespace sill {
  template class array_factor<double>;
  template class array_factor<float>;
}

using namespace arma;
using namespace sill;

struct iarray : public array_factor<int> {
  typedef array_factor<int> base;
  iarray() { }
  explicit iarray(const domain_type& args) { reset(args); }
  iarray(const domain_type& args, const array_type& param, bool zero_nan = false)
    : base(args, param, zero_nan) { }
  iarray(const domain_type& args, std::initializer_list<int> values) {
    reset(args);
    assert(size() == values.size());
    std::copy(values.begin(), values.end(), begin());
  }
  iarray(const domain_type& args, int value) {
    reset(args);
    param_.fill(value);
  }
  bool operator==(const iarray& other) const {
    return this->equal(other);
  }
  bool operator!=(const iarray& other) const {
    return !this->equal(other);
  }
  
  using base::join_inplace;
  using base::aggregate;
  using base::restrict;
  using base::restrict_join;
};

std::ostream& operator<<(std::ostream& out, const iarray& f) {
  out << f.arguments() << std::endl
      << f.param() << std::endl;
  return out;
}

typedef iarray::domain_type domain2;

BOOST_AUTO_TEST_CASE(test_join) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  iarray f({}, {3});
  iarray fx({x}, {1, 2});
  iarray fy({y}, {2, 3, 4});
  iarray fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray fyx({y, x}, {-1, -2, -3, 4, 5, 6});
  iarray g({}, {2});
  iarray gx({x}, {0, 1});
  iarray gy({y}, {1, 2, 3});
  iarray gxy({x, y}, {0, 1, 2, 3, 4, 5});
  iarray gyx({y, x}, gxy.param().transpose());
  
  iarray h;
  h = join<iarray>(f, g, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({}, {5}));
  BOOST_CHECK_NE(h, iarray({}, {6}));
  BOOST_CHECK_NE(h, iarray({x}, {1, 2}));

  h = join<iarray>(f, gx, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x}, {3, 4}));

  h = join<iarray>(fx, g, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x}, {3, 4}));

  h = join<iarray>(f, gxy, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x, y}, {3, 4, 5, 6, 7, 8}));

  h = join<iarray>(fxy, g, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x, y}, {1, 0, -1, 6, 7, 8}));

  h = join<iarray>(fx, gy, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x, y}, {2, 3, 3, 4, 4, 5}));

  h = join<iarray>(fxy, gx, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x, y}, {-1, -1, -3, 5, 5, 7}));

  h = join<iarray>(fxy, gy, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x, y}, {0, -1, -1, 6, 8, 9}));
  
  h = join<iarray>(fxy, gxy, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x, y}, {-1, -1, -1, 7, 9, 11}));

  h = join<iarray>(fxy, gyx, sill::plus<>());
  BOOST_CHECK_EQUAL(h, iarray({x, y}, {-1, -1, -1, 7, 9, 11}));
}

BOOST_AUTO_TEST_CASE(test_join_inplace) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  iarray f({}, {3});
  iarray fx({x}, {1, 2});
  iarray fy({y}, {2, 3, 4});
  iarray fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray fyx({y, x}, {-1, -2, -3, 4, 5, 6});
  iarray gx({x}, {0, 1});
  iarray gxy({x, y}, {0, 1, 2, 3, 4, 5});

  iarray hx;
  (hx = gx).join_inplace(f, sill::plus_assign<>(), false);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {3, 4}));

  (hx = gx).join_inplace(fx, sill::plus_assign<>(), false);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {1, 3}));

  iarray hxy;
  (hxy = gxy).join_inplace(f, sill::plus_assign<>(), false);
  BOOST_CHECK_EQUAL(hxy, iarray({x, y}, {3, 4, 5, 6, 7, 8}));
  
  (hxy = gxy).join_inplace(fx, sill::plus_assign<>(), false);
  BOOST_CHECK_EQUAL(hxy, iarray({x, y}, {1, 3, 3, 5, 5, 7}));

  (hxy = gxy).join_inplace(fy, sill::plus_assign<>(), false);
  BOOST_CHECK_EQUAL(hxy, iarray({x, y}, {2, 3, 5, 6, 8, 9}));

  (hxy = gxy).join_inplace(fxy, sill::plus_assign<>(), false);
  BOOST_CHECK_EQUAL(hxy, iarray({x, y}, {-1, -1, -1, 7, 9, 11}));

  (hxy = gxy).join_inplace(fyx, sill::plus_assign<>(), false);
  BOOST_CHECK_EQUAL(hxy, iarray({x, y}, {-1, 5, 0, 8, 1, 11}));
}

BOOST_AUTO_TEST_CASE(test_aggregate) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  iarray fxy({x, y}, {-1, -2, -3, 4, 5, 6});

  iarray h;
  fxy.aggregate({}, sum_op(), h);
  BOOST_CHECK_EQUAL(h, iarray({}, {9}));

  iarray hx;
  fxy.aggregate({x}, sum_op(), hx);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {1, 8}));

  iarray hy;
  fxy.aggregate({y}, sum_op(), hy);
  BOOST_CHECK_EQUAL(hy, iarray({y}, {-3, 1, 11}));

  iarray hxy;
  fxy.aggregate({x, y}, sum_op(), hxy);
  BOOST_CHECK_EQUAL(hxy, fxy);

  iarray hyx;
  fxy.aggregate({y, x}, sum_op(), hyx);
  BOOST_CHECK_EQUAL(hyx, iarray({y, x}, {-1, -3, 5, -2, 4, 6}));
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  finite_variable* z = u.new_finite_variable("z", 3);
  finite_assignment empty = {{z, 2}};

  iarray f({}, 1);
  iarray fy({y}, {2, 3, 4});
  iarray fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  
  iarray h;
  f.restrict(empty, h);
  BOOST_CHECK_EQUAL(h, iarray({}, 1));

  fy.restrict({{y, 1}}, h);
  BOOST_CHECK_EQUAL(h, iarray({}, 3));

  fxy.restrict({{x, 1}, {y, 2}}, h);
  BOOST_CHECK_EQUAL(h, iarray({}, 6));

  iarray hy;
  fy.restrict(empty, hy);
  BOOST_CHECK_EQUAL(hy, fy);

  fxy.restrict({{x, 1}}, hy);
  BOOST_CHECK_EQUAL(hy, iarray({y}, {-2, 4, 6}));

  iarray hx;
  fxy.restrict({{y, 2}}, hx);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {5, 6}));

  iarray hxy;
  fxy.restrict(empty, hxy);
  BOOST_CHECK_EQUAL(hxy, fxy);
}

BOOST_AUTO_TEST_CASE(test_restrict_join) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  finite_variable* z = u.new_finite_variable("z", 3);
  finite_assignment empty = {{z, 2}};

  iarray fy({y}, {2, 3, 4});
  iarray fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray fyx({y, x}, fxy.param().transpose());
  iarray gx({x}, {4, 2});
  iarray gy({y}, {1, 2, 0});

  // no restrict
  iarray hxy = fxy;
  fy.restrict_join(empty, sill::plus_assign<>(), false, hxy);
  BOOST_CHECK_EQUAL(hxy, iarray({x, y}, {1, 0, 0, 7, 9, 10}));

  // restrict a unary
  iarray hx = gx;
  fy.restrict_join({{y, 1}}, sill::plus_assign<>(), false, hx);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {7, 5}));

  // restrict a binary
  hx = gx;
  fxy.restrict_join({{x, 1}, {y, 2}}, sill::plus_assign<>(), false, hx);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {10, 8}));

  hx = gx;
  fxy.restrict_join({{y, 2}}, sill::plus_assign<>(), false, hx);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {9, 8}));

  hx = gx;
  fyx.restrict_join({{y, 2}}, sill::plus_assign<>(), false, hx);
  BOOST_CHECK_EQUAL(hx, iarray({x}, {9, 8}));

  iarray hy = gy;
  fxy.restrict_join({{x, 1}}, sill::plus_assign<>(), false, hy);
  BOOST_CHECK_EQUAL(hy, iarray({y}, {-1, 6, 6}));

  hy = gy;
  fyx.restrict_join({{x, 1}}, sill::plus_assign<>(), false, hy);
  BOOST_CHECK_EQUAL(hy, iarray({y}, {-1, 6, 6}));
}
