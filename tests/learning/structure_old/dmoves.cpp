#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset_old/data_conversions.hpp>
#include <sill/learning/dataset_old/dataset_statistics.hpp>
#include <sill/learning/dataset_old/syn_oracle_bayes_net.hpp>
#include <sill/learning/dataset_old/vector_dataset.hpp>
#include <sill/learning/structure_old/decomposable_iterators.hpp>
#include <sill/learning/structure_old/decomposable_search.hpp>
#include <sill/learning/structure_old/entropy_score.hpp>
#include <sill/learning/structure_old/dmove_move_leaf.hpp>
#include <sill/learning/structure_old/dmove_push_back_subtree.hpp>
#include <sill/learning/structure_old/dmove_swap_variables.hpp>
#include <sill/math/permutations.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file dmoves.cpp Test different decomposable moves for structure search.
 */

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Create a dataset to work with
  size_t n_records = 2000;
  size_t size_evidence = 2;
  size_t n = 5;
  size_t n_states = 2;
  size_t n_emissions = 2;

  universe u;
  unsigned random_seed = 69371053;
  unsigned random_seed2 = 940283949;
  boost::mt11213b rng(random_seed);

  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, n_states, n_emissions);

  syn_oracle_bayes_net<table_factor>::parameters bn_oracle_params;
  bn_oracle_params.random_seed = random_seed2;
  syn_oracle_bayes_net<table_factor> bn_oracle(bn, bn_oracle_params);
  vector_dataset_old<> ds;
  oracle2dataset(bn_oracle, n_records, ds);
  dataset_statistics<> stats(ds);

  // Create an initial model
  bool use_estimates = false;
  bool maximal_JTs = true;
  size_t max_clique_size = 3;

  learnt_decomposable<table_factor>::parameters ld_params;
  ld_params.maximal_JTs(maximal_JTs).max_clique_size(max_clique_size);
  star_decomposable_iterator<table_factor>::parameters star_it_params;
  star_it_params.max_clique_size(max_clique_size);
  star_decomposable_iterator<table_factor> star_it(stats, star_it_params);
  if (!(star_it.next()))
    assert(false);
  learnt_decomposable<table_factor> model(star_it.current(), ld_params);

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

  entropy_score<table_factor>::parameters score_params;
  score_params.evidence(evidence);
  entropy_score<table_factor> score(score_params);
  double init_score = score.compute(model);
  double cur_score(0);

  cout << "ORIGINAL MODEL:\n" << model << "\n"
       << "\t evidence = " << evidence << "\n"
       << "\t current score = " << cur_score << "\n" << endl;

  //! The type of queue used to store moves
  typedef mutable_queue<decomposable_change<table_factor>*, double> queue_type;
  queue_type change_queue;
  std::vector<decomposable_change<table_factor>::clique_change> clique_changes;
  learnt_decomposable<table_factor> tmpmodel;

  // DMOVE_MOVE_LEAF

  tmpmodel = model;
  cur_score = init_score;
  dmove_move_leaf<table_factor, queue_type> dmove_move_leaf_;
  dmove_move_leaf_.generate_all_moves(tmpmodel, cur_score, score, stats,
                                      change_queue, use_estimates);
  cout << "DMOVE_MOVE_LEAF: " << change_queue.size() << " moves in queue"
       << endl;
  if (!change_queue.empty()) {
    double score_change = change_queue.top().second;
    decomposable_change<table_factor>* move_ptr = change_queue.top().first;
    change_queue.pop();
    if (move_ptr->valid(tmpmodel))
      cout << "\t Move is valid.\n";
    else
      cout << "\t ERROR: Move is NOT valid.\n";
    clique_changes = move_ptr->commit(tmpmodel, stats);
    delete(move_ptr);
    cur_score += score_change;
    cout << "Committed top move with score change " << score_change << ":\n"
         << tmpmodel << endl;
    while(change_queue.size() > 0) {
      move_ptr = change_queue.top().first;
      change_queue.pop();
      delete(move_ptr);
    }
  }
  dmove_move_leaf_.generate_new_moves(tmpmodel, cur_score, score,
                                      clique_changes,
                                      stats, change_queue, use_estimates);
  cout << "Generated " << change_queue.size() << " new moves;\n"
       << " best new move has score_change " << change_queue.top().second
       << endl;
  while(change_queue.size() > 0) {
    decomposable_change<table_factor>* move_ptr = change_queue.top().first;
    change_queue.pop();
    delete(move_ptr);
  }
  cout << "Resetting model, clearing queue.\n"
       << "======================================================\n" << endl;

  // DMOVE_SWAP_VARIABLES

  tmpmodel = model;
  cur_score = init_score;
  dmove_swap_variables<table_factor, queue_type> dmove_swap_variables_;
  dmove_swap_variables_.generate_all_moves(tmpmodel, cur_score, score, stats,
                                      change_queue, use_estimates);
  cout << "DMOVE_SWAP_VARIABLES: " << change_queue.size() << " moves in queue"
       << endl;
  if (!change_queue.empty()) {
    double score_change = change_queue.top().second;
    decomposable_change<table_factor>* move_ptr = change_queue.top().first;
    change_queue.pop();
    if (move_ptr->valid(tmpmodel))
      cout << "\t Move is valid.\n";
    else
      cout << "\t ERROR: Move is NOT valid.\n";
    clique_changes = move_ptr->commit(tmpmodel, stats);
    delete(move_ptr);
    cur_score += score_change;
    cout << "Committed top move with score change " << score_change << ":\n"
         << tmpmodel << endl;
    while(change_queue.size() > 0) {
      move_ptr = change_queue.top().first;
      change_queue.pop();
      delete(move_ptr);
    }
  }
  dmove_swap_variables_.generate_new_moves(tmpmodel, cur_score, score,
                                           clique_changes,
                                           stats, change_queue, use_estimates);
  cout << "Generated " << change_queue.size() << " new moves;\n"
       << " best new move has score_change " << change_queue.top().second
       << endl;
  while(change_queue.size() > 0) {
    decomposable_change<table_factor>* move_ptr = change_queue.top().first;
    change_queue.pop();
    delete(move_ptr);
  }
  cout << "Resetting model, clearing queue.\n"
       << "======================================================\n" << endl;

  // DMOVE_PUSH_BACK_SUBTREE

  tmpmodel = model;
  cur_score = init_score;
  dmove_push_back_subtree<table_factor, queue_type> dmove_push_back_subtree_;
  dmove_push_back_subtree_.generate_all_moves(tmpmodel, cur_score, score, stats,
                                      change_queue, use_estimates);
  cout << "DMOVE_PUSH_BACK_SUBTREE: " << change_queue.size() << " moves in queue"
       << endl;
  if (!change_queue.empty()) {
    double score_change = change_queue.top().second;
    decomposable_change<table_factor>* move_ptr = change_queue.top().first;
    change_queue.pop();
    if (move_ptr->valid(tmpmodel))
      cout << "\t Move is valid.\n";
    else
      cout << "\t ERROR: Move is NOT valid.\n";
    clique_changes = move_ptr->commit(tmpmodel, stats);
    delete(move_ptr);
    cur_score += score_change;
    cout << "Committed top move with score change " << score_change << ":\n"
         << tmpmodel << endl;
    while(change_queue.size() > 0) {
      move_ptr = change_queue.top().first;
      change_queue.pop();
      delete(move_ptr);
    }
  }
  dmove_push_back_subtree_.generate_new_moves(tmpmodel, cur_score, score,
                                              clique_changes, stats,
                                              change_queue, use_estimates);
  cout << "Generated " << change_queue.size() << " new moves;\n"
       << " best new move has score_change " << change_queue.top().second
       << endl;
  while(change_queue.size() > 0) {
    decomposable_change<table_factor>* move_ptr = change_queue.top().first;
    change_queue.pop();
    delete(move_ptr);
  }
  cout << "Resetting model, clearing queue.\n"
       << "======================================================\n" << endl;

  return EXIT_SUCCESS;
}
