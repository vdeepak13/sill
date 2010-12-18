#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/random.hpp>

int main() {

  using namespace sill;
  using namespace std;

  size_t n = 10;
  size_t n_states = 4;
  size_t n_emissions = 4;

  universe u;
  unsigned random_seed = 69371053;
  boost::mt11213b rng(random_seed);

  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, n_states, n_emissions);
  cout << "Random HMM:\n" << bn << endl;

}
