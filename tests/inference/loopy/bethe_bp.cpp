#define BOOST_TEST_MODULE bethe_bp
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/random/moment_gaussian_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/special/grid_graph.hpp>
#include <sill/inference/loopy/asynchronous_bethe_bp.hpp>
#include <sill/inference/loopy/residual_bethe_bp.hpp>
#include <sill/model/markov_network.hpp>

#include <sill/macros_def.hpp>

// TODO: deal with cliques larger than 2

namespace sill {
  template class asynchronous_bethe_bp<canonical_gaussian>;
  template class residual_bethe_bp<canonical_gaussian>;
  
  template class asynchronous_bethe_bp<table_factor>;
  template class residual_bethe_bp<table_factor>;
}

using namespace sill;

void test(bethe_bp<canonical_gaussian>* engine,
          size_t niters,
          const moment_gaussian& joint,
          double tol) {
  for (size_t it = 0; it < niters; ++it) {
    engine->iterate(1.0);
  }

  // check that the marginal means have converged to the true means
  double maxerror = 0.0;
  foreach(vector_variable* v, joint.arguments()) {
    moment_gaussian belief(engine->belief(make_domain(v)));
    double error = std::abs(belief.mean()[0] - joint.mean(v)[0]);
    maxerror = std::max(maxerror, error);
    BOOST_CHECK_LE(error, tol);
  }
  
  std::cout << "Maximum error: " << maxerror << std::endl;

  delete engine;
}

BOOST_AUTO_TEST_CASE(test_convergence) {
  size_t m = 5;
  size_t n = 4;

  // construct a grid network with attractive Gaussian potentials
  universe u;
  vector_var_vector variables = u.new_vector_variables(m*n, 1);
  pairwise_markov_network<canonical_gaussian> model;
  make_grid_graph(variables, m, n, model);

  moment_gaussian_generator gen;
  boost::mt19937 rng;
  foreach(undirected_edge<vector_variable*> e, model.edges()) {
    model[e] = canonical_gaussian(gen(model.nodes(e), rng));
  }
  moment_gaussian joint(prod_all(model.factors()));
  
  test(new asynchronous_bethe_bp<canonical_gaussian>(model), 10, joint, 1e-5);
  test(new residual_bethe_bp<canonical_gaussian>(model), m*n*50, joint, 1e-5);
}
