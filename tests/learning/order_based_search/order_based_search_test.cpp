/**
 * \file order_based_search.cpp Test of Order Based Search for Bayes nets
 *
 * Uses code from Koller and Teyssier.  This code uses statistics files, rather
 * than the datasets used by other parts of PRL.
 *
 * \todo Replace their code with our own--this should be reasonably easy since
 *       it will largely involve replacing their implementations of factors, etc.
 *       with our own.
 */

#include <iostream>

#include <cmath>
#include <sill/datastructure/dense_table.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/inference/belief_propagation.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/order_based_search/Classes.hpp>
#include <sill/learning/order_based_search/Liste.hpp>
#include <sill/learning/order_based_search/constants.hpp>
#include <sill/learning/order_based_search/Bayesian.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/pairwise_mn_conversion.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char* argv[]) {

  if (argc != 5) {
    std::cout << "Usage: train-dataset test-dataset maxparents timelimit"
              << std::endl;
    return 1;
  }

  std::string train_ds_file(argv[1]);
  std::string test_ds_file(argv[2]);
  int maxparents = atoi(argv[3]);
  int timelimit = atoi(argv[4]);
  int search_depth = 180;
  int deviation = 37;
  int tabu_size = 180;
  int rmin = 8;
  int ncand = UNDEF;
  size_t niters = 100;

  universe u;

  boost::shared_ptr<vector_dataset<> > train_ds_ptr
    = data_loader::load_symbolic_dataset<vector_dataset<> >(train_ds_file, u);
  vector_dataset<>& train_ds = *train_ds_ptr;
  boost::shared_ptr<vector_dataset<> > test_ds_ptr
    = data_loader::load_symbolic_dataset<vector_dataset<> >(test_ds_file,
                                              train_ds.var_order());
  vector_dataset<>& test_ds = *test_ds_ptr;

  PopulateCombin();
  bayesian_graph<> bg =
    GreedyAnalyse( train_ds, test_ds,  timelimit,  search_depth,  deviation,
                   tabu_size, rmin, ncand, maxparents);
  bayesian_network<tablef> bn(bg);
  double smoothing = .001 / .999;
  foreach(variable_h v, bn.arguments()) {
    domain vin(bn.parents(v));
    tablef f = train_ds.marginal<tablef>(vin.plus(v), smoothing);
    tablef f2 = f.marginal(vin);
    f /= f2;
    bn.factor(v) = f;
  }

  std::cout << "Learned model:" << std::endl
            << bn << std::endl;

  std::cout << "Now, convert to pairwise Markov net:" << std::endl;
  typedef pairwise_markov_network<tablef> pmn_type;
  pmn_type pmn;
  std::map<variable_h, std::vector<sill::variable_h> > var_mapping;
  boost::tie(pmn, var_mapping) = fm2pairwise_markov_network(bn, u);
  std::cout << pmn << std::endl;
/*
  std::cout << "Now, use BP to compute some stuff..." << std::endl;

  residual_loopy_bp<pmn_type> bp_engine(pmn);
  for (size_t i = 0; i < niters; i++) {
    bp_engine.iterate(1);
    std::cout << "ITERATION" << i << " - vertex beliefs:\n";
    foreach(pmn_type::vertex_descriptor v, pmn.vertices()) {
      std::cout << bp_engine.belief(v);
    }
    std::cout << std::endl;
  }
*/

  return 0; 

}
