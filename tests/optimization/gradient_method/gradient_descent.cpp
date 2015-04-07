#include <sill/optimization/gradient_method/gradient_descent.hpp>
#include <sill/optimization/line_search/backtracking_line_search.hpp>

#include <boost/bind.hpp>

#include "../quadratic_objective.hpp"

using namespace sill;

int main() {
  quadratic_objective objective(vec2(2, 3), mat22(2, 1, 1, 2));
  line_search<vec_type>* search = new backtracking_line_search<vec_type>;
  gradient_descent<vec_type> gd(search);
  gd.objective(&objective);
  gd.solution(vec2(0, 0));
  for (size_t it = 0; !gd.converged(); ++it) {
    line_search_result<double> result = gd.iterate();
    std::cout << "Iteration " << it << ", result " << result << std::endl;
  }
  std::cout << "Computed solution: " << gd.solution().transpose();
}
