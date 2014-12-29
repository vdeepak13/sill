#include <sill/optimization/gradient_method/conjugate_gradient.hpp>
#include <sill/optimization/line_search/backtracking_line_search.hpp>
#include <sill/optimization/line_search/slope_binary_search.hpp>

#include <boost/bind.hpp>

#include "../quadratic_objective.hpp"

using namespace sill;

int main() {
  quadratic_objective objective("2 3", "2 1; 1 2");
  //line_search<vec_type>* search = new backtracking_line_search<vec_type>;
  line_search<vec_type>* search = new slope_binary_search<vec_type>;
  conjugate_gradient<vec_type> gd(search);
  gd.objective(&objective);
  gd.solution("0 0");
  for (size_t it = 0; !gd.converged(); ++it) {
    line_search_result<double> result = gd.iterate();
    std::cout << "Iteration " << it << ", result " << result << std::endl;
  }
  std::cout << "Computed solution: " << gd.solution().t();
}
