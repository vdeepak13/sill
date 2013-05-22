
#include <iostream>

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/optimization/lbfgs.hpp>

using namespace sill;

struct obj_functor {
  double objective(vec val) const {
    vec tmpval = vec_2(1.,1.);
    return -5. + dot(val - tmpval, val - tmpval);
  }
};

struct grad_functor {
  void gradient(vec& grad, const vec& val) const {
    vec tmpval = vec_2(1.,1.);
    grad = 2. * (val - tmpval);
  }
};

int main(int argc, char* argv[]) {

  size_t niter = 5;

  vec val = vec_2(0.,0.);
  lbfgs_parameters lbfgs_params;
  lbfgs_params.debug = 2;
  obj_functor obj;
  grad_functor grad;
  lbfgs<vec, obj_functor, grad_functor>
    lbfgs(obj, grad, val, lbfgs_params);
  std::cerr << "Iteration\tObjective\tChange\tx" << std::endl;
  for (size_t t(0); t < niter; ++t) {
    if (!lbfgs.step())
      break;
    std::cerr << lbfgs.iteration() << "\t" << lbfgs.objective() << "\t"
              << lbfgs.objective_change() << "\t" << lbfgs.x() << std::endl;
  }
  std::cerr << "Final values:\n"
            << lbfgs.iteration() << "\t" << lbfgs.objective() << "\t"
            << lbfgs.objective_change() << "\t" << lbfgs.x() << std::endl;
  return 0;

}

