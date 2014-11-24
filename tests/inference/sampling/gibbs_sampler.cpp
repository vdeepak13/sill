#define BOOST_TEST_MODULE gibbs_sampler
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/functional.hpp>
#include <sill/factor/random/ising_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/inference/sampling/gibbs_sampler.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

double get_error(const std::vector<table_factor>& var_marginals,
                 const std::vector<table_factor>& approx_var_marginals) {
  assert(var_marginals.size() == approx_var_marginals.size());
  double error = 0.0;
  for (size_t j = 0; j < var_marginals.size(); ++j) {
    table_factor marginal = approx_var_marginals[j];
    error += norm_1(var_marginals[j], marginal.normalize());
  }
  return error / var_marginals.size();
}

BOOST_AUTO_TEST_CASE(test_convergence) {
  using namespace std;

  size_t m = 4;
  size_t n = 4;

  size_t nsamples = 500000;
  unsigned random_seed = 2390249;

  universe u;
  boost::mt19937 rng(random_seed);
  ising_factor_generator gen(0.0, 0.5, 0.0, 1.0);
  
  finite_var_vector variables = u.new_finite_variables(m*n, 2);
  pairwise_markov_network<table_factor> mn;
  make_grid_graph(variables, m, n, mn);
  mn.initialize(marginal_fn(gen, rng));
  cout << "Generated random " << m << " x " << n << " Markov network."
       << endl;

  std::vector<table_factor> var_marginals;
  decomposable<table_factor> joint;
  joint *= mn.factors();
  foreach(finite_variable* v, variables) {
    var_marginals.push_back(joint.marginal(make_domain(v)));
  }
  cout << "Computed exact marginals." << endl;

  cout << "Samples\tAvg L1 Error" << endl;
  sequential_gibbs_sampler<table_factor> sampler(mn);
  std::vector<table_factor> approx_var_marginals;
  foreach(finite_variable* v, variables) {
    approx_var_marginals.push_back(table_factor(make_domain(v), 0));
  }

  std::vector<std::pair<size_t, double> > avg_L1_errors;
  for (size_t i = 0; i < nsamples; ++i) {
    const finite_record_old& sample = sampler.next_sample();
    for (size_t j = 0; j < variables.size(); ++j) {
      approx_var_marginals[j](sample)++;
    }
    if ((i+1) % (nsamples / 10) == 0) {
      cout << i << "\t" << get_error(var_marginals, approx_var_marginals) << endl;
    }
  }

  BOOST_CHECK_LE(get_error(var_marginals, approx_var_marginals), 0.01);
}
