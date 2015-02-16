#define BOOST_TEST_MODULE array_factor
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/base/array_factor.hpp>
#include <sill/functional/operators.hpp>
#include <sill/functional/eigen.hpp>

namespace sill {
  template class array_factor<double, 1>;
  template class array_factor<double, 2>;
}

using namespace sill;

template <size_t N>
struct iarray : public array_factor<int, N> {
  typedef array_factor<int, N> base;
  typedef typename base::domain_type domain_type;
  typedef typename base::array_type array_type;
  iarray() { }
  explicit iarray(const domain_type& args) { this->reset(args); }
  iarray(const domain_type& args, const array_type& param)
    : base(args, param) { }
  iarray(const domain_type& args, std::initializer_list<int> values)
    : base(args, values) { }
  bool operator==(const iarray& other) const { return this->equal(other); }
  bool operator!=(const iarray& other) const { return !this->equal(other); }
  friend std::ostream&
  operator<<(std::ostream& out, const iarray& f) {
    out << f.arguments() << std::endl
        << f.param() << std::endl;
    return out;
  }
};

typedef iarray<1> iarray1;
typedef iarray<2> iarray2;

BOOST_AUTO_TEST_CASE(test_join) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  iarray1 fx({x}, {1, 2});
  iarray1 fy({y}, {2, 3, 4});
  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fyx({y, x}, {-1, -2, -3, 4, 5, 6});
  iarray1 gx({x}, {0, 1});
  iarray1 gy({y}, {1, 2, 3});
  iarray2 gxy({x, y}, {0, 1, 2, 3, 4, 5});
  iarray2 gyx({y, x}, gxy.param().transpose());
  
  iarray1 h1;
  iarray2 h2;
  h1 = join<iarray1>(fx, gx, sill::plus<>());
  BOOST_CHECK_EQUAL(h1, iarray1({x}, {1, 3}));

  //h = join<iarray>(fx, gy, sill::plus<>());
  //BOOST_CHECK_EQUAL(h, iarray({x, y}, {2, 3, 3, 4, 4, 5}));

  h2 = join<iarray2>(fxy, gx, sill::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -3, 5, 5, 7}));

  h2 = join<iarray2>(fxy, gy, sill::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {0, -1, -1, 6, 8, 9}));
  
  h2 = join<iarray2>(gx, fxy, sill::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -3, 5, 5, 7}));

  h2 = join<iarray2>(gy, fxy, sill::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({y, x}, {0, -1, 8, -1, 6, 9}));
  
  h2 = join<iarray2>(fxy, gxy, sill::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -1, 7, 9, 11}));

  h2 = join<iarray2>(fxy, gyx, sill::plus<>());
  BOOST_CHECK_EQUAL(h2, iarray2({x, y}, {-1, -1, -1, 7, 9, 11}));
}

BOOST_AUTO_TEST_CASE(test_join_inplace) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  iarray1 fx({x}, {1, 2});
  iarray1 fy({y}, {2, 3, 4});
  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fyx({y, x}, {-1, -2, -3, 4, 5, 6});
  iarray1 gx({x}, {0, 1});
  iarray2 gxy({x, y}, {0, 1, 2, 3, 4, 5});

  iarray1 hx;
  join_inplace(hx = gx, fx, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {1, 3}));

  iarray2 hxy;
  join_inplace(hxy = gxy, fx, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {1, 3, 3, 5, 5, 7}));

  join_inplace(hxy = gxy, fy, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {2, 3, 5, 6, 8, 9}));

  join_inplace(hxy = gxy, fxy, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {-1, -1, -1, 7, 9, 11}));

  join_inplace(hxy = gxy, fyx, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hxy, iarray2({x, y}, {-1, 5, 0, 8, 1, 11}));
}

BOOST_AUTO_TEST_CASE(test_aggregate) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);

  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});

  iarray1 hx;
  aggregate(fxy, {x}, hx, sum_op());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {1, 8}));

  iarray1 hy;
  aggregate(fxy, {y}, hy, sum_op());
  BOOST_CHECK_EQUAL(hy, iarray1({y}, {-3, 1, 11}));
}

BOOST_AUTO_TEST_CASE(test_restrict) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  finite_variable* z = u.new_finite_variable("z", 3);
  finite_assignment empty = {{z, 2}};

  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  
  iarray1 hy;
  restrict_assign(fxy, {{x, 1}}, hy);
  BOOST_CHECK_EQUAL(hy, iarray1({y}, {-2, 4, 6}));

  iarray1 hx;
  restrict_assign(fxy, {{y, 2}}, hx);
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {5, 6}));
}

BOOST_AUTO_TEST_CASE(test_restrict_join) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  finite_variable* z = u.new_finite_variable("z", 2);
  finite_assignment empty = {{z, 1}};

  iarray1 fy({y}, {2, 3, 4});
  iarray2 fxy({x, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fzy({z, y}, {-1, -2, -3, 4, 5, 6});
  iarray2 fyx({y, x}, fxy.param().transpose());
  iarray1 gx({x}, {4, 2});
  iarray1 gy({y}, {1, 2, 0});

  iarray1 hx;
  hx = gx;
  restrict_join(fxy, {{x, 1}, {y, 2}}, hx, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {9, 8}));

  hx = gx;
  restrict_join(fxy, {{y, 2}}, hx, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {9, 8}));

  hx = gx;
  restrict_join(fzy, {{z, 1}, {y, 2}}, hx, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hx, iarray1({x}, {10, 8}));

  iarray1 hy = gy;
  restrict_join(fxy, {{x, 1}}, hy, sill::plus_assign<>());
  BOOST_CHECK_EQUAL(hy, iarray1({y}, {-1, 6, 6}));
}
