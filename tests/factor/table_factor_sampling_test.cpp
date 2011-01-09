#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file table_factor_sampling_test.cpp  Test sampling for table factors.
 *
 * Generate a table_factor and compare:
 *  - the true entropy
 *  - the cross entropy between the sample distribution and the truth
 */

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Dataset parameters
  size_t nsamples = 1000000;
  size_t nvars = 2;
  size_t arity = 4;
  unsigned random_seed = time(NULL);

  universe u;
  finite_domain vars;
  for (size_t i = 0; i < nvars; ++i)
    vars.insert(u.new_finite_variable(arity));
  boost::mt11213b rng(random_seed);

  // Create a model to sample from.
  table_factor f(random_discrete_factor<table_factor>(vars, rng));
  f.normalize();

  // Test log likelihood.
  double true_entropy = f.entropy();
  double cross_entropy = 0;
  cout << "Random table factor with " << nvars << " vars with arity "
       << arity << ".\n"
       << "Sampling and comparing true entropy with cross entropy:\n"
       << "nsamples\tTruth\tX entropy\tDifference\n"
       << "---------------------------------------------------" << endl;
  for (size_t i(0); i < nsamples; ++i) {
    finite_assignment a(f.sample(rng));
    cross_entropy -= f.logv(a);
    if ((i+1) % (nsamples / 10) == 0) {
      double estimate(cross_entropy / (i+1.));
      cout << (i+1) << "\t\t" << true_entropy << "\t" << estimate << "\t"
           << (true_entropy - estimate) << endl;
    }
  }

  // Test conditioning and computing log likelihoods...TO DO

  return 0;
}

#include <sill/macros_undef.hpp>
