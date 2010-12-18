#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file decomposable_sampling_test.cpp  Test sampling.
 *
 * Generate a decomposable model with table factors and compare:
 *  - the true entropy of the model
 *  - the cross entropy between the sample distribution and the truth
 */

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Dataset parameters
  size_t nsamples = 10000;
  size_t n = 50; // length of width-2 chain decomposable model
  unsigned oracle_seed = 1284392;

  universe u;
  boost::mt11213b rng(oracle_seed);

  // Create a model to sample from
  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, 2, 2);
  decomposable<table_factor> model(bn.factors());

  // Test conditioning and computing log likelihoods.
  finite_domain half_vars1(model.arguments());
  finite_domain half_vars2;
  for (size_t i(0); i < n / 2; ++i) {
    assert(half_vars1.size() > 0);
    finite_variable* v = *(half_vars1.begin());
    half_vars1.erase(v);
    half_vars2.insert(v);
  }
  decomposable<table_factor> half_vars1_model;
  model.marginal(half_vars1, half_vars1_model);

  // Sample
  double true_entropy(model.entropy());
  double cross_entropy(0);
  double ll_half_vars1(0);
  double ll_half_vars2_given_1(0);
  cout << "Made tabular decomposable model.\n"
       << "Sampling and comparing true entropy with cross entropy:\n"
       << "samples\tTruth\tX entropy\tDifference\tLoglike(V1)\tLoglike(V2|V1)\tSum\n"
       << "---------------------------------------------------" << endl;
  for (size_t i(0); i < nsamples; ++i) {
    finite_assignment a(model.sample(rng));
    cross_entropy -= model.log_likelihood(a);
    decomposable<table_factor> conditioned_model(model);
    finite_assignment a_half_vars1(map_intersect(a, half_vars1));
    conditioned_model.condition(a_half_vars1);
    ll_half_vars1 += half_vars1_model.log_likelihood(a);
    ll_half_vars2_given_1 += conditioned_model.log_likelihood(a);
    if (i % (nsamples / 10) == 0) {
      double estimate(cross_entropy / (i+1.));
      double e1(ll_half_vars1 / (i+1.));
      double e2(ll_half_vars2_given_1 / (i+1.));
      double e3((ll_half_vars1 + ll_half_vars2_given_1) / (i+1.));
      cout << (i+1) << "\t" << true_entropy << "\t" << estimate << "\t"
           << (true_entropy - estimate) << "\t" << e1 << "\t" << e2 << "\t"
	   << e3 << endl;
    }
  }

  return 0;
}
