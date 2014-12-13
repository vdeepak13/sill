#define BOOST_TEST_MODULE mean_field_bipartite
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_matrix.hpp>
#include <sill/factor/probability_matrix.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/bipartite_graph.hpp>
#include <sill/inference/exact/junction_tree_inference.hpp>
#include <sill/inference/variational/mean_field_bipartite.hpp>

#include <boost/random/uniform_int_distribution.hpp>

using namespace sill;

// consider boost strong typedef
template <int index>
struct vertex { 
  finite_variable* v;
  vertex() : v(NULL) { }
  vertex(finite_variable* v) : v(v) { }
  size_t id() const { return v ? v->id() : -1; }
  bool operator<(vertex u) const { return id() < u.id(); }
  bool operator==(vertex u) const { return id() == u.id(); }
  bool operator!=(vertex u) const { return id() != u.id(); }
};

template <int index>
size_t hash_value(vertex<index> v) {
  return v.id();
}

typedef vertex<1> vertex1;
typedef vertex<2> vertex2;

BOOST_AUTO_TEST_CASE(test_convergence) {
  size_t nvertices = 20;
  size_t nedges = 50;
  size_t niters = 20;

  // Create a random model
  universe u;
  uniform_factor_generator gen;
  boost::mt19937 rng;
  bipartite_graph<vertex1, vertex2, canonical_matrix<>, canonical_matrix<> > model;
  std::vector<table_factor> factors;
  finite_var_vector vars1;
  finite_var_vector vars2;

  // node potentials
  for (size_t i = 0; i < nvertices; ++i) {
    finite_variable* v1 = u.new_finite_variable("x", 2);
    finite_variable* v2 = u.new_finite_variable("y", 2);
    vars1.push_back(v1);
    vars2.push_back(v2); 
    table_factor f1 = gen(make_domain(v1), rng);
    table_factor f2 = gen(make_domain(v2), rng);
    factors.push_back(f1);
    factors.push_back(f2);
    model.add_vertex(vertex1(v1), canonical_matrix<>(f1));
    model.add_vertex(vertex2(v2), canonical_matrix<>(f2));
  }

  // edge potentials
  boost::random::uniform_int_distribution<size_t> unif(0, nvertices - 1);
  for (size_t i = 0; i < nedges; ++i) {
    finite_variable* v1 = vars1[unif(rng)];
    finite_variable* v2 = vars2[unif(rng)];
    table_factor f = gen(make_domain(v1, v2), rng);
    factors.push_back(f);
    model.add_edge(vertex1(v1), vertex2(v2), canonical_matrix<>(f));
  }

  // run exact inference
  shafer_shenoy<table_factor> exact(factors);
  std::cout << "Tree width of the model: " << exact.tree_width() << std::endl;
  exact.calibrate();
  exact.normalize();
  std::cout << "Finished exact inference" << std::endl;

  // run mean field inference
  mean_field_bipartite<vertex1, vertex2, canonical_matrix<> > mf(&model, 4);
  double diff;
  for (size_t it = 0; it < niters; ++it) {
    diff = mf.iterate();
    std::cout << "Iteration " << it << ": " << diff << std::endl;
  }
  BOOST_CHECK_LT(diff, 1e-4);
  
  // compute the KL divergence from exact to mean field
  double kl1 = 0.0;
  double kl2 = 0.0;
  for (size_t i = 0; i < nvertices; ++i) {
    probability_matrix<> exact1(exact.belief(make_domain(vars1[i])));
    probability_matrix<> exact2(exact.belief(make_domain(vars2[i])));
    kl1 += exact1.relative_entropy(mf.belief(vertex1(vars1[i])));
    kl2 += exact2.relative_entropy(mf.belief(vertex2(vars2[i])));
  }
  kl1 /= nvertices;
  kl2 /= nvertices;
  std::cout << "Average kl1 = " << kl1 << std::endl;
  std::cout << "Average kl2 = " << kl2 << std::endl;
  BOOST_CHECK_LT(kl1, 0.02);
  BOOST_CHECK_LT(kl2, 0.02);
}

