#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/syn_oracle_bayes_net.hpp>
#include <sill/learning/dataset/syn_oracle_majority.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  size_t nsamples = 5;

  // Create a Bayes net
  size_t n = 1;
  size_t n_states = 2;
  size_t n_emissions = 2;

  universe u;
  unsigned random_seed = 69371053;
  boost::mt11213b rng(random_seed);

  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, n_states, n_emissions);
  cout << "Sampling from Bayes net:\n" << bn << endl;

  syn_oracle_bayes_net<table_factor> bn_oracle(bn);
  for (size_t i = 0; i < nsamples; ++i) {
    bn_oracle.next();
    cout << "\t" << bn_oracle.current().assignment() << endl;
  }

}
