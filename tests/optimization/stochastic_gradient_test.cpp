
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/optimization/stochastic_gradient.hpp>

// minimize -5 + (val - <1,1>)^2
struct obj_grad_functor1 {
  explicit obj_grad_functor1(double range = 1)
    : unif_real(0,range), rng(time(NULL)) { }
  double objective(sill::vec val) const {
    const sill::vec v1("1 1");
    return -5. + dot(val - v1, val - v1);
  }
  void gradient(sill::vec& grad, const sill::vec& val) const {
    const sill::vec v1("1 1");
    grad = 2. * (val - v1);
    double gradL2 = grad.L2norm();
    for (size_t i = 0; i < v1.size(); ++i) {
      grad[i] += unif_real(rng) * gradL2;
    }
  }
  mutable boost::uniform_real<double> unif_real;
  mutable boost::mt11213b rng;
};

int main(int argc, char* argv[]) {

  using namespace sill;

  size_t niter = 100;
  double range = 1; // controls amount of noise added to the gradient

  stochastic_gradient_parameters sg_params;
  sg_params.single_opt_step_params.set_shrink_eta(niter);

  std::cerr << "sg_params:\n" << sg_params << std::endl;

  obj_grad_functor1 og1(range);
  vec val1(2, 0);
  stochastic_gradient<vec, obj_grad_functor1>
    sg1(og1, val1, sg_params);
  double last_obj = og1.objective(sg1.x());
  std::cout << "Testing stochastic gradient\n"
            << "-------------------------------------------------------\n"
            << "Iteration\tObjective\tChange\tx" << std::endl;
  for (size_t t(0); t < niter; ++t) {
    if (!sg1.step())
      break;
    if (t % (niter / 10) == 0) {
      double current_obj = og1.objective(sg1.x());
      std::cout << sg1.iteration() << "\t\t" << current_obj << "\t\t"
                << (current_obj - last_obj) << "\t" << sg1.x() << std::endl;
      last_obj = current_obj;
    }
  }
  double current_obj = og1.objective(sg1.x());
  std::cout << sg1.iteration() << "\t\t" << current_obj << "\t\t"
            << (current_obj - last_obj) << "\t" << sg1.x() << std::endl;

  return 0;

}

