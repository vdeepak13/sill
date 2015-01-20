#define BOOST_TEST_MODULE vector_record
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/vector_record.hpp>

namespace sill {
  // TODO: once we templatize assignments, add specialization for float
  template class vector_record<double>;
}

using namespace sill;

BOOST_AUTO_TEST_CASE(test_extract) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 2);
  vector_variable* z = u.new_vector_variable("z", 1);
  vector_var_vector vars = make_vector(x, y, z);

  vector_record<> r(vars);
  BOOST_CHECK_EQUAL(r.values.size(), 4);
  r.values = "1 2 0.5 -0.5";

  vector_assignment a;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.size(), 3);
  BOOST_CHECK(equal(a[x], vec("1")));
  BOOST_CHECK(equal(a[y], vec("2 0.5")));
  BOOST_CHECK(equal(a[z], vec("-0.5")));

  r.values[0] = std::numeric_limits<double>::quiet_NaN();
  r.values[3] = 0.1;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.size(), 2);
  BOOST_CHECK_EQUAL(a.count(x), 0);
  BOOST_CHECK(equal(a[y], vec("2 0.5")));
  BOOST_CHECK(equal(a[z], vec("0.1")));
}

BOOST_AUTO_TEST_CASE(test_count_missing) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 2);
  vector_variable* z = u.new_vector_variable("z", 1);
  vector_var_vector vars = make_vector(x, y, z);
  vector_var_vector subseq = make_vector(x, z);

  vector_record<> r(vars);
  BOOST_CHECK_EQUAL(r.values.size(), 4);
  r.values = "1 2 0.5 -0.5";
  BOOST_CHECK_EQUAL(r.count_missing(), 0);
  BOOST_CHECK_EQUAL(r.count_missing(subseq), 0);

  r.values[0] = std::numeric_limits<double>::quiet_NaN();
  r.values[3] = std::numeric_limits<double>::quiet_NaN();
  BOOST_CHECK_EQUAL(r.count_missing(), 2);
  BOOST_CHECK_EQUAL(r.count_missing(subseq), 2);
  BOOST_CHECK_EQUAL(r.count_missing(make_vector(x, y)), 1);
}
