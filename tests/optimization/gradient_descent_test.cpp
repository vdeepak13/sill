
#include <iostream>

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/optimization/gradient_descent.hpp>

// minimize -5 + (val - <1,1>)^2
struct obj_grad_functor1 {
  double objective(sill::vec val) const {
    const sill::vec v1("1 1");
    return -5. + dot(val - v1, val - v1);
  }
  void gradient(sill::vec& grad, const sill::vec& val) const {
    const sill::vec v1("1 1");
    grad = 2. * (val - v1);
  }
};

int main(int argc, char* argv[]) {

  using namespace sill;

  size_t niter = 5;

  gradient_descent_parameters cg_params;

  obj_grad_functor1 og1;
  vec val1(2, 0);
  gradient_descent<vec, obj_grad_functor1, obj_grad_functor1>
    cg1(og1, og1, val1, cg_params);
  std::cerr << "Test 1\n"
            << "-------------------------------------------------------"
            << "Iteration\tObjective\tChange\tx" << std::endl;
  for (size_t t(0); t < niter; ++t) {
    if (!cg1.step())
      break;
    std::cerr << cg1.iteration() << "\t\t" << cg1.objective() << "\t\t"
              << cg1.objective_change() << "\t" << cg1.x() << std::endl;
  }
  std::cerr << "Final values:\n"
            << cg1.iteration() << "\t\t" << cg1.objective() << "\t\t"
            << cg1.objective_change() << "\t" << cg1.x() << std::endl;

  return 0;

}

