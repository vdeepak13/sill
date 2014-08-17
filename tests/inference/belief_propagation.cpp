#define BOOST_TEST_MODULE belief_propagation
#include <boost/test/unit_test.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/random/moment_gaussian_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/inference/belief_propagation.hpp>
#include <sill/model/markov_network.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
typedef pairwise_markov_network<canonical_gaussian> cg_model_type;
typedef pairwise_markov_network<table_factor> tf_model_type;

template class synchronous_loopy_bp<cg_model_type>;
template class asynchronous_loopy_bp<cg_model_type>;
template class residual_loopy_bp<cg_model_type>;

template class synchronous_loopy_bp<tf_model_type>;
template class asynchronous_loopy_bp<tf_model_type>;
template class residual_loopy_bp<tf_model_type>;

void test(loopy_bp_engine<cg_model_type>* engine,
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
  const cg_model_type& gm = engine->graphical_model();
  foreach(cg_model_type::vertex v, gm.vertices()) {
    canonical_gaussian nbelief = engine->belief(v);
    foreach(cg_model_type::edge e, gm.in_edges(v)) {
      canonical_gaussian ebelief = engine->belief(e).marginal(make_domain(v));
      BOOST_CHECK_LE(norm_inf(nbelief, ebelief), error);
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
  cg_model_type model;
  make_grid_graph(variables, m, n, model);
  
  moment_gaussian_generator gen;
  boost::mt19937 rng;
  foreach(cg_model_type::edge e, model.edges()) {
    model[e] = canonical_gaussian(gen(model.nodes(e), rng));
  }
  moment_gaussian joint(prod_all(model.factors()));
  
  test(new synchronous_loopy_bp<cg_model_type>(model), 10, joint, 1e-5);
  test(new asynchronous_loopy_bp<cg_model_type>(model), 10, joint, 1e-5);
  test(new residual_loopy_bp<cg_model_type>(model), m*n*10, joint, 1e-5);
}

