
#include <boost/timer.hpp>

#include <sill/learning/chow_liu.hpp>
#include <sill/learning/dataset/assignment_dataset.hpp>
#include <sill/learning/dataset/generate_datasets.hpp>
#include <sill/learning/dataset/syn_oracle_bayes_net.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * Time Chow-Liu implementations:
 *  - Implementation using factors2thin_decomposable
 * (This was originally used to compare multiple implementations.)
 *
 * \author Joseph Bradley
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  // Generate some training and test data (from a random HMM)
  size_t n = 3;
  size_t n_states = 2;
  size_t n_emissions = 2;
  size_t ntrain = 1000;
  size_t ntest = 1000;
  unsigned random_seed = 69371053;
  unsigned o_random_seed = 120592;

  universe u;
  boost::mt11213b rng(random_seed);

  bayesian_network<table_factor> bn;
  random_HMM(bn, rng, u, n, n_states, n_emissions);

  std::cout << bn << std::endl;

  syn_oracle_bayes_net<table_factor>::parameters syn_oracle_params;
  syn_oracle_params.random_seed = o_random_seed;
  syn_oracle_bayes_net<table_factor> bn_oracle(bn);
  assignment_dataset<> train_ds;
  oracle2dataset(bn_oracle, ntrain, train_ds);
  assignment_dataset<> test_ds;
  oracle2dataset(bn_oracle, ntest, test_ds);

  // Time creation of Chow-Liu trees.
  size_t ntrials = 1;
  double tmp = 0;
  boost::timer t;

  // -- Implementation using factors2thin_decomposable --
  t.restart();
  for (size_t n(0); n < ntrials; ++n) {
    cout << "doing trial " << n << std::endl;
    chow_liu<table_factor> chowliu(train_ds.finite_variables(),
                                   train_ds);
    const decomposable<table_factor>& model = chowliu.model();
    cout << model << std::endl;
    if (argc == 1000)
      tmp += model.arguments().size();
    if (n + 1 == ntrials) {
      cout << " implementation using factors2thin_decomposable: "
           << t.elapsed() / ntrials << std::endl;
      double ll(0);
      foreach(const record<>& rec, test_ds.records())
        ll += model.log_likelihood(rec);
      ll /= test_ds.size();
      cout << "   avg test log likelihood = " << ll << std::endl;
    }
  }

  if (argc == 1000)
    cout << tmp;

}
