#define BOOST_TEST_MODULE gradient_descent
#include <boost/test/unit_test.hpp>

#include <sill/optimization/gradient_method/gradient_descent.hpp>
#include <sill/optimization/line_search/backtracking_line_search.hpp>

#include "../quadratic_objective.hpp"

namespace sill {
  template class gradient_descent<vec_type>;
}

using namespace sill;

BOOST_AUTO_TEST_CASE(test_convergence) {
  quadratic_objective objective(vec2(2, 3), mat22(2, 1, 1, 2));
  line_search<vec_type>* search = new backtracking_line_search<vec_type>;
  gradient_descent<vec_type> gd(search);
  gd.objective(&objective);
  gd.solution(vec2(0, 0));
  for (size_t it = 0; it < 20 && !gd.converged(); ++it) {
    line_search_result<double> result = gd.iterate();
    std::cout << "Iteration " << it << ", result " << result << std::endl;
  }
  std::cout << "Estimate: " << gd.solution().transpose() << std::endl;
  BOOST_CHECK(gd.converged());
  BOOST_CHECK_SMALL((gd.solution() - objective.ctr).norm(), 1e-3);
}
