
#include <iostream>

#include <sill/math/vector.hpp>
#include <sill/optimization/line_search_with_grad.hpp>

// minimize -5 + (val - 1)^2
class obj_functor
  : public sill::real_opt_step_functor {

  double objective(double val) const {
    return - 5 + (val - 1.) * (val - 1.);
  }
  double gradient(double val) const {
    return 2 * (val - 1);
  }
  bool stop_early() const {
    return false;
  }
  bool valid_gradient() const {
    return true;
  }
};

int main(int argc, char* argv[]) {

  using namespace sill;

  double init_val(.1);
  obj_functor obj;
  line_search ls1;
  ls1.get_params().ls_init_eta = .1;
  ls1.get_params().convergence_zero = .000001;
  ls1.get_params().debug = 3;
  ls1.step(obj);
  std::cout << "line_search with quadratic function f(x) = -5 + (x-1)^2"
            << std::endl;
  std::cout << " Search with objective only:\n"
            << "   " << ls1.bounding_steps() << " bounding steps, "
            << ls1.searching_steps() << " searching steps\n"
            << "   Returned x = " << ls1.eta() << " which gives value "
            << ls1.objective() << "\n" << std::endl;

  line_search_with_grad ls2;
  ls2.get_params().ls_init_eta = .1;
  ls2.get_params().convergence_zero = .000001;
  ls2.get_params().debug = 3;
  ls2.step(obj);
  std::cout << " Search with objective and gradient:\n"
            << "   " << ls2.bounding_steps() << " bounding steps, "
            << ls2.searching_steps() << " searching steps\n"
            << "   Returned x = " << ls2.eta() << " which gives value "
            << ls2.objective() << std::endl;
  return 0;

}
