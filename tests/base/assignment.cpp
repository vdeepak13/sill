#define BOOST_TEST_MODULE assignment
#include <boost/test/unit_test.hpp>

#include <boost/array.hpp>

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_assignment_iterator.hpp>
#include <sill/base/vector_assignment.hpp>
#include <sill/base/universe.hpp>

using namespace sill;

BOOST_TEST_DONT_PRINT_LOG_VALUE(sill::finite_assignment_iterator);

struct fixture {
  fixture()
    : v(u.new_finite_variable(2)),
      x(u.new_finite_variable(3)),
      y(u.new_vector_variable(2)) { }
  
  universe u;
  finite_variable* v;
  finite_variable* x;
  vector_variable* y;
};

BOOST_FIXTURE_TEST_CASE(test_assignments, fixture) {
  // Create an assignment.
  finite_assignment fa;
  vector_assignment va;
  fa[x] = 2;
  // va[y] = vector_variable::value_type(2); 
  // warning: this only initializes the dimensions, not the content
  va[y] = zeros(2);

  BOOST_CHECK_EQUAL(fa[x], 2);
  BOOST_CHECK(accu(va[y] == zeros(2)) == 2);
}

BOOST_FIXTURE_TEST_CASE(test_finite_assignment_iterator, fixture) {
  // Test the assignment iterator
  boost::array<finite_variable*, 2> d = {{ v, x }};
  finite_assignment_iterator it(d), end;
  finite_assignment fa;
  for (size_t i = 0; i < 3; ++i) {
    fa[x] = i;
    for (size_t j = 0; j < 2; ++j) {
      fa[v] = j;
      BOOST_CHECK_NE(it, end);
      BOOST_CHECK_EQUAL(*it++, fa);
    }
  }
  BOOST_CHECK_EQUAL(it, end);

  // Test the empty iterator
  boost::array<finite_variable*, 0> empty;
  it = finite_assignment_iterator(empty);
  fa.clear();
  BOOST_CHECK_EQUAL(*it, fa);
  BOOST_CHECK_EQUAL(++it, end);
}
