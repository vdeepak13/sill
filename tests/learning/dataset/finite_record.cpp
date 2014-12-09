#define BOOST_TEST_MODULE finite_record
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/finite_record.hpp>

using namespace sill;

BOOST_AUTO_TEST_CASE(test_extract) {
  universe u;
  finite_var_vector vars = u.new_finite_variables(3, 4);
  finite_record r(vars);
  r.values[0] = 1;
  r.values[1] = 2;
  r.values[2] = 3;

  finite_assignment a;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.size(), 3);
  BOOST_CHECK_EQUAL(a[vars[0]], 1);
  BOOST_CHECK_EQUAL(a[vars[1]], 2);
  BOOST_CHECK_EQUAL(a[vars[2]], 3);

  r.values[0] = 0;
  r.values[1] = -1;
  r.extract(a);
  BOOST_CHECK_EQUAL(a.size(), 2);
  BOOST_CHECK_EQUAL(a[vars[0]], 0);
  BOOST_CHECK_EQUAL(a[vars[2]], 3);
  BOOST_CHECK_EQUAL(a.count(vars[1]), 0);
}

BOOST_AUTO_TEST_CASE(test_count_missing) {
  universe u;
  finite_var_vector vars = u.new_finite_variables(4, 4);
  finite_var_vector subseq = make_vector(vars[1], vars[3]);

  finite_record r(vars);
  r.values[0] = 1;
  r.values[1] = 0;
  r.values[2] = 2;
  r.values[3] = 3;
  
  BOOST_CHECK_EQUAL(r.count_missing(), 0);
  BOOST_CHECK_EQUAL(r.count_missing(subseq), 0);

  r.values[1] = -1;
  BOOST_CHECK_EQUAL(r.count_missing(), 1);
  BOOST_CHECK_EQUAL(r.count_missing(subseq), 1);
}
