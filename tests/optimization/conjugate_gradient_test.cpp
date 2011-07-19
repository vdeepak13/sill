
#include <iostream>

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/optimization/conjugate_gradient.hpp>

using namespace sill;

static const double CGT_OBJ_CONST = 1;

struct obj_functor {
  double objective(vec val) const {
    vec tmpval = vec_2(1.,1.);
    return -5. +
      CGT_OBJ_CONST * dot(val - tmpval, val - tmpval);
  }
};

struct grad_functor {
  void gradient(vec& grad, const vec& val) const {
    vec tmpval = vec_2(1.,1.);
    grad = 2. * CGT_OBJ_CONST * (val - tmpval);
  }
};

struct prec_functor {
  void precondition(vec& grad, const vec& val) const {
    grad *= 1. / CGT_OBJ_CONST;
  }
  void precondition(vec& grad) const {
    grad *= 1. / CGT_OBJ_CONST;
  }
};

int main(int argc, char* argv[]) {

  size_t niter = 5;

  vec val = vec_2(0.,0.);
  conjugate_gradient_parameters cg_params;
  cg_params.debug = 2;
  obj_functor obj;
  grad_functor grad;
  conjugate_gradient<vec, obj_functor, grad_functor>
    cg(obj, grad, val, cg_params);
  std::cout << "Iteration\tObjective\tChange\tx" << std::endl;
  for (size_t t(0); t < niter; ++t) {
    if (!cg.step())
      break;
    std::cout << cg.iteration() << "\t" << cg.objective() << "\t"
              << cg.objective_change() << "\t" << cg.x() << std::endl;
  }
  std::cout << "Final values:\n"
            << cg.iteration() << "\t" << cg.objective() << "\t"
            << cg.objective_change() << "\t" << cg.x() << std::endl;

  std::cout << "============================================================\n"
            << "Now testing using a preconditioner..." << std::endl;

  prec_functor prec;
  val = 0.;
  conjugate_gradient<vec, obj_functor, grad_functor, prec_functor>
    cg2(obj, grad, prec, val, cg_params);
  std::cout << "Iteration\tObjective\tChange\tx" << std::endl;
  for (size_t t(0); t < niter; ++t) {
    if (!cg2.step())
      break;
    std::cout << cg2.iteration() << "\t" << cg2.objective() << "\t"
              << cg2.objective_change() << "\t" << cg2.x() << std::endl;
  }
  std::cout << "Final values:\n"
            << cg2.iteration() << "\t" << cg2.objective() << "\t"
            << cg2.objective_change() << "\t" << cg2.x() << std::endl;

  return 0;
}

