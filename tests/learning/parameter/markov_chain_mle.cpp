#define BOOST_TEST_MODULE markov_chain_mle
#include <boost/test/unit_test.hpp>

#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/finite_sequence_record.hpp>
#include <sill/learning/dataset/vector_sequence_record.hpp>
#include <sill/learning/dataset/sequence_memory_dataset.hpp>
#include <sill/learning/dataset/vector_sequence_record.hpp>
#include <sill/learning/parameter/markov_chain_mle.hpp>

using namespace sill;

template class sill::markov_chain<moment_gaussian>;
template class sill::markov_chain<table_factor>;

template class sill::markov_chain_mle<moment_gaussian>;
template class sill::markov_chain_mle<table_factor>;

BOOST_AUTO_TEST_CASE(test_reconstruct) {
  size_t nchains = 1000;
  size_t length = 100;

  // create the processes
  dprocess x = new vector_discrete_process("x", 1);
  dprocess y = new vector_discrete_process("y", 2);
  std::vector<dprocess> xy = make_vector(x, y);
  domain xy_t = variables(xy, current_step);
  domain xy_t1 = variables(xy, next_step);

  // create the chain
  moment_gaussian init(xy_t, "-1 0 1", "2 0.5 0.2; 0.5 1 0.2; 0.2 0.2 1");
  moment_gaussian tran(xy_t1, "0.1 0.2 0.3", arma::eye(3, 3),
                       xy_t, "1 1 0; 0 1 1; 0 0 1");
  markov_chain<moment_gaussian> chain(init, tran);

  // draw random samples from the chain
  sequence_memory_dataset<vector_dataset<> > ds;
  ds.initialize(xy);
  vector_sequence_record<> r(xy);
  boost::lagged_fibonacci607 rng;
  for (size_t i = 0; i < nchains; ++i) {
    chain.sample(length, r, rng);
    ds.insert(r);
  }

  // estimate a chain from the samples
  markov_chain_mle<moment_gaussian> learner(xy);
  markov_chain<moment_gaussian> lchain;
  double ll = learner.learn(ds, lchain);
  
  // print out the stats
  std::cout << "Log-likelihood: " << ll << std::endl;
  std::cout << "Ground truth: " << init << tran << std::endl;
  std::cout << "Estimate: " << lchain.initial() << lchain.transition() << std::endl;
}
