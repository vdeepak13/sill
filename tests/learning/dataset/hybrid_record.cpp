#define BOOST_TEST_MODULE hybrid_record
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/hybrid_record.hpp>

using namespace sill;

// TODO: once we templatize assignments, add specialization for float
template class hybrid_record<double>;

BOOST_AUTO_TEST_CASE(test_extract) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 2);
  vector_variable* z = u.new_vector_variable("z", 1);
  finite_var_vector finite_vars = u.new_finite_variables(4, 3);
  vector_var_vector vector_vars = make_vector(x, y, z);

  hybrid_record<> r(finite_vars, vector_vars);
  BOOST_CHECK_EQUAL(r.values.finite.size(), 4);
  BOOST_CHECK_EQUAL(r.values.vector.size(), 4);
  r.values.finite[0] = 1;
  r.values.finite[1] = 2;
  r.values.finite[2] = 0;
  r.values.finite[3] = 1;
  r.values.vector = "1 2 0.5 -0.5";

  assignment a;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.finite().size(), 4);
  BOOST_CHECK_EQUAL(a.vector().size(), 3);
  BOOST_CHECK_EQUAL(a[finite_vars[0]], 1);
  BOOST_CHECK_EQUAL(a[finite_vars[1]], 2);
  BOOST_CHECK_EQUAL(a[finite_vars[2]], 0);
  BOOST_CHECK_EQUAL(a[finite_vars[3]], 1);
  BOOST_CHECK(equal(a[x], vec("1")));
  BOOST_CHECK(equal(a[y], vec("2 0.5")));
  BOOST_CHECK(equal(a[z], vec("-0.5")));

  r.values.finite[2] = -1;
  r.values.finite[3] = -1;
  r.values.vector[0] = std::numeric_limits<double>::quiet_NaN();
  r.values.vector[3] = std::numeric_limits<double>::quiet_NaN();
  r.extract(a);
  BOOST_CHECK_EQUAL(a.finite().size(), 2);
  BOOST_CHECK_EQUAL(a.vector().size(), 1);
  BOOST_CHECK_EQUAL(a[finite_vars[0]], 1);
  BOOST_CHECK_EQUAL(a[finite_vars[1]], 2);
  BOOST_CHECK(equal(a[y], vec("2 0.5")));
}

BOOST_AUTO_TEST_CASE(test_count_missing) {
  universe u;
  vector_variable* x = u.new_vector_variable("x", 1);
  vector_variable* y = u.new_vector_variable("y", 2);
  vector_variable* z = u.new_vector_variable("z", 1);
  finite_var_vector finite_vars = u.new_finite_variables(4, 3);
  vector_var_vector vector_vars = make_vector(x, y, z);
  var_vector subseq = make_vector<variable>(finite_vars[1], x, y, finite_vars[3]);

  hybrid_record<> r(finite_vars, vector_vars);
  BOOST_CHECK_EQUAL(r.values.finite.size(), 4);
  BOOST_CHECK_EQUAL(r.values.vector.size(), 4);
  r.values.finite[0] = 1;
  r.values.finite[1] = 2;
  r.values.finite[2] = 0;
  r.values.finite[3] = 1;
  r.values.vector = "1 2 0.5 -0.5";
  BOOST_CHECK_EQUAL(r.count_missing(), 0);
  BOOST_CHECK_EQUAL(r.count_missing(subseq), 0);

  r.values.finite[1] = -1;
  r.values.vector[1] = std::numeric_limits<double>::quiet_NaN();
  r.values.vector[3] = std::numeric_limits<double>::quiet_NaN();
  BOOST_CHECK_EQUAL(r.count_missing(), 3);
  BOOST_CHECK_EQUAL(r.count_missing(subseq), 2);
  BOOST_CHECK_EQUAL(r.count_missing(make_vector<variable>(x, y)), 1);
}
