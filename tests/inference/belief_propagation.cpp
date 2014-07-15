#define BOOST_TEST_MODULE belief_propagation
#include <boost/test/unit_test.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/random/random_canonical_gaussian_functor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/inference/belief_propagation.hpp>
#include <sill/model/markov_network.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
typedef pairwise_markov_network<canonical_gaussian> model_type;

void test(loopy_bp_engine<model_type>* engine,
          size_t niters,
          const moment_gaussian& joint,
          double error) {
  engine->iterate(niters);

  // check that the marginal means have converged to the true means
  foreach(vector_variable* v, joint.arguments()) {
    moment_gaussian belief(engine->belief(v));
    BOOST_CHECK_LE(std::abs(belief.mean()[0] - joint.mean(v)[0]), error);
  }

  // check that the edge marginals agree on the shared variable
  const model_type& gm = engine->graphical_model();
  foreach(model_type::vertex v, gm.vertices()) {
    canonical_gaussian nbelief = engine->belief(v);
    foreach(model_type::edge e, gm.in_edges(v)) {
      canonical_gaussian ebelief = engine->belief(e).marginal(make_domain(v));
      BOOST_CHECK_LE(std::abs(norm_inf(nbelief, ebelief)), error);
    }
  }
  delete engine;
}

BOOST_AUTO_TEST_CASE(test_convergence) {
  size_t m = 5;
  size_t n = 4;

  // construct a grid network with attractive Gaussian potentials
  universe u;
  vector_var_vector variables = u.new_vector_variables(m*n, 1);
  model_type model;
  make_grid_graph(variables, m, n, model);
  random_canonical_gaussian_functor gen;
  foreach(model_type::edge e, model.edges()) {
    model[e] = gen.generate_marginal(model.nodes(e));
  }
  moment_gaussian joint(prod_all(model.factors()));
  
  test(new synchronous_loopy_bp<model_type>(model), 10, joint, 1e-5);
  test(new asynchronous_loopy_bp<model_type>(model), 10, joint, 1e-5);
  test(new residual_loopy_bp<model_type>(model), m*n*10, joint, 1e-5);
}

