#define BOOST_TEST_MODULE backtracking_line_search
#include <boost/test/unit_test.hpp>

#include <sill/optimization/line_search/backtracking_line_search.hpp>

#include <boost/bind.hpp>

#include "../quadratic_objective.hpp"

using namespace sill;
typedef line_search_result<double> result_type;

template class backtracking_line_search<vec_type>;

BOOST_AUTO_TEST_CASE(test_exponential_decay_search) {
  quadratic_objective objective("5 4", "1 0; 0 1");
  backtracking_line_search_parameters<double> params(0.3, 0.5);
  backtracking_line_search<vec_type> search(params);
  search.reset(boost::bind(&quadratic_objective::value, &objective, _1),
               boost::bind(&quadratic_objective::gradient, &objective, _1));
  
  result_type r = search.step("1 2", "1 0.5");
  BOOST_CHECK_CLOSE(r.step, 1.0, 1e-6);
  BOOST_CHECK_CLOSE(r.value, 5.625, 1e-6);
}
