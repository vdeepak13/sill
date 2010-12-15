#include <iostream>
//#include <iterator>

#include <boost/random/mersenne_twister.hpp>

#include <prl/base/universe.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/learning/dataset/data_conversions.hpp>
#include <prl/learning/dataset/statistics.hpp>
#include <prl/learning/dataset/syn_oracle_bayes_net.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/structure_search/decomposable_iterators.hpp>
#include <prl/model/random.hpp>

using namespace prl;

// Test of decomposable iterators.
int main() {

  using namespace prl;
  using namespace std;

  // Create a dataset to work with
  size_t n_records = 2000;
  size_t n = 5;
  size_t n_states = 2;
  size_t n_emissions = 2;
  bool maximal_JTs = true;
  size_t max_clique_size = 3;

  universe u;
  unsigned random_seed = 69371053;
  boost::mt11213b rng(random_seed);

  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, n_states, n_emissions);

  learnt_decomposable<table_factor>::parameters ld_params;
  ld_params.maximal_JTs(maximal_JTs).max_clique_size(max_clique_size);
  learnt_decomposable<table_factor> model(bn.factors(), ld_params);

  syn_oracle_bayes_net<table_factor> bn_oracle(bn);
  boost::shared_ptr<vector_dataset>
    ds_ptr(oracle2dataset<vector_dataset>(bn_oracle,n_records));
  statistics stats(*ds_ptr);

  // CREATE_EMPTY_DECOMPOSABLE

  decomposable<table_factor> empty_model;
  create_empty_decomposable<table_factor>(empty_model, bn.arguments(), stats);
  cout << "CREATE_EMPTY_DECOMPOSABLE: \n" << empty_model
       << "======================================================\n" << endl;

  // STAR_DECOMPOSABLE_ITERATOR

  star_decomposable_iterator<table_factor>::parameters star_it_params;
  star_it_params.max_clique_size(max_clique_size);
  star_decomposable_iterator<table_factor> star_it(stats, star_it_params);
  if (!(star_it.next()))
    assert(false);
  cout << "STAR_DECOMPOSABLE_ITERATOR:\n" << star_it.current() << endl;
  size_t star_it_cnt = 1;
  while(star_it.next())
    ++star_it_cnt;
  cout << " star_decomposable_iterator generated " << star_it_cnt
       << " models in total.\n"
       << "======================================================\n" << endl;

}
