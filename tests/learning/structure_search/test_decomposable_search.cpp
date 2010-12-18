#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/structure_search/decomposable_search.hpp>
#include <sill/learning/structure_search/entropy_score.hpp>
#include <sill/learning/structure_search/dmove_push_back_subtree.hpp>
#include <sill/math/free_functions.hpp>
#include <sill/model/random.hpp>

#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/statistics.hpp>
#include <sill/learning/dataset/syn_oracle_bayes_net.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Create a dataset to work with
  size_t n_records = 2000;
  size_t size_evidence = 4;
  size_t n = 10;
  size_t n_states = 4;
  size_t n_emissions = 4;

  universe u;
  unsigned random_seed = 69371053;
  boost::mt11213b rng(random_seed);

  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, n_states, n_emissions);

  syn_oracle_bayes_net<table_factor> bn_oracle(bn);
  boost::shared_ptr<vector_dataset>
    ds_ptr(oracle2dataset<vector_dataset>(bn_oracle,n_records));
  statistics stats(*ds_ptr);

  // Choose some random variables for evidence
  finite_assignment evidence;
  finite_domain args(bn.arguments());
  std::vector<size_t> permutation(randperm(args.size(), rng));
  finite_var_vector arg_vector(args.begin(), args.end());
  for (size_t i = 0; i < size_evidence; ++i) {
    finite_variable* var = arg_vector[permutation[i]];
    boost::uniform_int<int> uniform_int(0, var->size() - 1);
    evidence[var] = uniform_int(rng);
  }

  // Learn the model from the dataset
  size_t n_steps = 2;
  bool maximal_JTs = true;
  size_t max_clique_size = 3;
  entropy_score<table_factor>::parameters score_params;
  score_params.evidence(evidence);
  entropy_score<table_factor> score(score_params);
  std::vector<decomposable_search<table_factor>::decomposable_move_type*>
    allowed_moves;
  allowed_moves.push_back
    (new dmove_push_back_subtree
     <table_factor, decomposable_search<table_factor>::queue_type>());
  decomposable<table_factor> initial_model(bn.factors());
  decomposable_search<table_factor>::parameters search_params;
  search_params.maximal_JTs(maximal_JTs).max_clique_size(max_clique_size);
  decomposable_search<table_factor> search(bn.arguments(), stats, score,
                                     allowed_moves, initial_model,
                                     search_params);
  for (size_t i = 0; i < n_steps; ++i) {
    if (!(search.step()))
      break;
  }

  return EXIT_SUCCESS;
}
