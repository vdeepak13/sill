#define BOOST_TEST_MODULE value_binary_search
#include <boost/test/unit_test.hpp>

#include <sill/optimization/line_search/value_binary_search.hpp>

#include <boost/bind.hpp>

#include "../quadratic_objective.hpp"

using namespace sill;
typedef line_search_result<double> result_type;

template class value_binary_search<vec_type>;

BOOST_AUTO_TEST_CASE(test_value_binary_search) {
  quadratic_objective objective("5 4", "1 0; 0 1");
  value_binary_search<vec_type> search;
  search.reset(boost::bind(&quadratic_objective::value, &objective, _1), NULL);
  result_type horiz = search.step("3.987 3", "1 0");
  BOOST_CHECK_CLOSE(horiz.step, 1.013, 1e-3);
  BOOST_CHECK_CLOSE(horiz.value, 0.5, 1e-3);
  result_type diag = search.step("1 0", "1 1");
  BOOST_CHECK_CLOSE(diag.step, 4.0, 1e-3);
  BOOST_CHECK_SMALL(diag.value, 1e-5);
}
